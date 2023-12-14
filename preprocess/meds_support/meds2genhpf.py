import os
from pathlib import Path
from argparse import ArgumentParser
import h5py
import numpy as np
import pandas as pd
import datasets
from tqdm import tqdm

import transforms


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=".")
    parser.add_argument("--output_path", type=str, default="outputs")
    parser.add_argument("--output_name", type=str, default="esds_genhpf_data")
    
    parser.add_argument("--max_num_events", type=int, default=256)
    parser.add_argument("--min_num_events", type=int, default=5)
    parser.add_argument("--max_tokens_per_event", type=int, default=192)
    parser.add_argument("--max_sequence_length", type=int, default=8192)

    return parser

def main(args):

    ###########################################################################
    # Load ESDS-compliant dataset
    # Note that any dataset can be loaded here as long as it follows ESDS schema.
    # By default, we load an example ESDS dataset extracted from EventStreamGPT.

    # Note that split names should be one of ["train", "valid", "test"] for GenHPF
    split_map = {"train": "train", "tuning": "valid", "held_out": "test"}
    ds = datasets.load_dataset(
        "parquet",
        data_files={
            split_map[sp]: str(Path(args.dataset_path) / sp / "*.parquet")
            for sp in ("train", "tuning", "held_out")
        }
    )
    ###########################################################################

    T = transforms.ToHierarchicalGenHPFStyle(
        exclude_codes=["gender", "event_type"],
        max_num_events=args.max_num_events,
        return_type_ids=True,
        return_dpe_ids=True
    )
    genhpf_hierarchical_ds = ds.map(T, batched=True, num_proc=32)


    T = transforms.ToFlattenedGenHPFStyle(
        exclude_codes=["gender", "event_type"],
        max_sequence_length=args.max_sequence_length,
        return_type_ids=True,
        return_dpe_ids=True
    )

    genhpf_flattened_ds = ds.map(T, batched=True, num_proc=32)

    if not os.path.exists(os.path.join(args.output_path)):
        os.makedirs(os.path.join(args.output_path))
    pids = {}
    with h5py.File(str(Path(args.output_path) / (args.output_name + ".h5")), "w") as f:
        ehr = f.create_group("ehr")
        for sp in tqdm(genhpf_hierarchical_ds, total=len(genhpf_flattened_ds)):
            pids[sp] = []
            for i, patient_id in tqdm(enumerate(genhpf_hierarchical_ds[sp]["patient_id"]), total=len(genhpf_flattened_ds[sp])):
                if genhpf_hierarchical_ds[sp][i]["events"] is None:
                    print(f"{patient_id} skipped (None events)")
                    continue

                if len(genhpf_hierarchical_ds[sp][i]["events"]) < args.min_num_events:
                    print(f"{patient_id} skipped (< 5 events)")
                    continue

                pids[sp].append(patient_id)
                stay = ehr.create_group(str(patient_id))

                # process hierarchical structure
                events = genhpf_hierarchical_ds[sp][i]["events"]
                input_ids = [x["measurements"]["input_ids"] for x in events]
                type_ids = [x["measurements"]["token_type_ids"] for x in events]
                dpe_ids = [x["measurements"]["dpe_ids"] for x in events]

                sizes = [len(x) for x in input_ids]
                collated_input_ids = np.zeros((len(input_ids), 1, args.max_tokens_per_event), dtype=int)
                collated_type_ids = np.zeros((len(input_ids), 1, args.max_tokens_per_event), dtype=int)
                collated_dpe_ids = np.zeros((len(input_ids), 1, args.max_tokens_per_event), dtype=int)

                for j, size in enumerate(sizes):
                    diff = size - args.max_tokens_per_event
                    if diff == 0:
                        collated_input_ids[j][0] = input_ids[j]
                        collated_type_ids[j][0] = type_ids[j]
                        collated_dpe_ids[j][0] = dpe_ids[j]
                    elif diff < 0:
                        collated_input_ids[j][0] = (
                            np.concatenate([input_ids[j], np.zeros((-diff,))]).astype(int)
                        )
                        collated_type_ids[j][0] = (
                            np.concatenate([type_ids[j], np.zeros((-diff,))]).astype(int)
                        )
                        collated_dpe_ids[j][0] = (
                            np.concatenate([dpe_ids[j], np.zeros((-diff,))]).astype(int)
                        )
                    else:
                        collated_input_ids[j][0] = np.concatenate(
                            [input_ids[j,:args.max_tokens_per_event-1], [input_ids[j, -1]]]
                        ).astype(int)
                
                hierarchical_data = np.concatenate(
                    [collated_input_ids, collated_type_ids, collated_dpe_ids], axis=1
                )
                stay.create_dataset("hi", data=hierarchical_data, dtype="i2", compression="lzf", shuffle=True)

                # process time
                start_time = genhpf_hierarchical_ds[sp][i]["icustay_start_time"]
                times = np.array([x["time"] for x in events])
                elapsed_times = np.array(
                    [round(x.total_seconds() / 60) for x in (times - start_time)],
                    dtype=int
                )
                stay.create_dataset("time", data=elapsed_times, dtype="i")

                # process flattened structure
                events = genhpf_flattened_ds[sp][i]["events"]
                input_ids = events["input_ids"]
                type_ids = events["token_type_ids"]
                dpe_ids = events["dpe_ids"]

                diff = len(input_ids) - args.max_sequence_length
                if diff == 0:
                    collated_input_ids = np.array(input_ids, dtype=int)
                    collated_type_ids = np.array(type_ids, dtype=int)
                    collated_dpe_ids = np.array(dpe_ids, dtype=int)
                elif diff < 0:
                    collated_input_ids = (
                        np.concatenate([input_ids, np.zeros((-diff,))]).astype(int)
                    )
                    collated_type_ids = (
                        np.concatenate([type_ids, np.zeros((-diff,))]).astype(int)
                    )
                    collated_dpe_ids = (
                        np.concatenate([dpe_ids, np.zeros((-diff,))]).astype(int)
                    )
                else:
                    # we handle for this case already in the process of transformation (Flatten())
                    raise ValueError()

                flattened_data = np.vstack(
                    [collated_input_ids, collated_type_ids, collated_dpe_ids]
                )
                stay.create_dataset("fl", data=flattened_data, dtype="i2", compression="lzf", shuffle=True)

    cohort = {"icustay_id": [], "split_1": []}
    for sp, patient_ids in pids.items():
        cohort["icustay_id"].extend(patient_ids)
        cohort["split_1"].extend([sp] * len(patient_ids))
    pd.DataFrame.from_dict(cohort).to_csv(os.path.join(args.output_path, args.output_name + ".csv"), index=False)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)