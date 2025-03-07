import argparse
import os

import h5py
import torch
from torch.nn.functional import one_hot
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

"""
Sample Dataset for Integrated-EHR-Pipeline
NOTE: The preprocessed token indices are stored as np.int16 type for efficienty,
        so overflow can be caused when vocab size > 32767.
 - ehr.cohort.labeled.index: dataframe pickle
    - index: should be resetted by `df.reset_index()`
    - [hi_start:hi_end] is the range of indices for the patient's history
    - split: one of ['train', 'val', 'test']
 - ehr.data: .np file with np.int16 type, (num_total_events, 3, max_word_len)
    - second dimension
        - 0: input_ids
        - 1: type_ids
        - 2: dpe_ids
"""


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ehr",
        type=str,
        required=True,
        choices=["mimiciii", "mimiciv", "eicu"],
        help="Name of the EHR dataset",
    )
    parser.add_argument("--data", type=str, required=True, help="Path to the preprocessed data")
    parser.add_argument(
        "--pred_target",
        type=str,
        required=True,
        choices=[
            "mortality",
            "readmission",
            "los_3day",
            "los_7day",
            "final_acuity",
            "imminent_discharge",
            "diagnosis",
        ],
        help="Prediction target",
    )
    parser.add_argument("--max_event_size", type=int, default=256, help="max event size to crop to")
    parser.add_argument(
        "--max_event_token_len", type=int, default=128, help="max token length for each event (Hierarchical)"
    )

    parser.add_argument(
        "--max_patient_token_len", type=int, default=8192, help="max token length for each patient (Flatten)"
    )
    return parser


class BaseEHRDataset(Dataset):
    def __init__(self, args, split):
        super().__init__()

        self.args = args
        self.data_path = os.path.join(args.data, f"{args.ehr}.h5")

        self.data = h5py.File(self.data_path, "r")["ehr"]
        self.keys = []
        for key in self.data.keys():
            if self.data[key].attrs["split"] == split:
                self.keys.append(key)
        self.pred_target = args.pred_target

        self.num_classes = {
            "mimiciii": {
                "mortality": 2,
                "readmission": 2,
                "los_3day": 2,
                "los_7day": 2,
                "final_acuity": 18,
                "imminent_discharge": 18,
                "diagnosis": 18,
            },
            "mimiciv": {
                "mortality": 2,
                "readmission": 2,
                "los_3day": 2,
                "los_7day": 2,
                "final_acuity": 14,
                "imminent_discharge": 14,
                "diagnosis": 18,
            },
            "eicu": {
                "mortality": 2,
                "readmission": 2,
                "los_3day": 2,
                "los_7day": 2,
                "final_acuity": 9,
                "imminent_discharge": 9,
                "diagnosis": 18,
            },
        }[self.args.ehr][self.args.pred_target]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        raise NotImplementedError()

    def collate_fn(self, out):
        ret = dict()
        if len(out) == 1:
            for k, v in out[0].items():
                if k == "label":
                    ret[k] = (
                        one_hot(torch.LongTensor([v]), self.num_classes)
                        .float()
                        .reshape((1, self.num_classes))
                    )
                else:
                    ret[k] = v.unsqueeze(0)
        else:
            for k, v in out[0].items():
                if k == "label":
                    ret[k] = one_hot(torch.LongTensor([i[k] for i in out]), self.num_classes).float()
                else:
                    ret[k] = pad_sequence([i[k] for i in out], batch_first=True)
        return ret


class HierarchicalEHRDataset(BaseEHRDataset):
    def __init__(self, args, split):
        super().__init__(args, split)

    def __getitem__(self, idx):
        # NOTE: Warning occurs when converting np.int16 read-only array into tensor, but ignoreable

        data = self.data[self.keys[idx]]["hi"]
        label = self.data[self.keys[idx]].attrs[self.pred_target]
        return {
            "input_ids": torch.IntTensor(data[:, 0, :]),
            "type_ids": torch.IntTensor(data[:, 1, :]),
            "dpe_ids": torch.IntTensor(data[:, 2, :]),
            "label": label,
        }


class FlattenEHRDataset(BaseEHRDataset):
    def __init__(self, args, split):
        super().__init__(args, split)
        self.data_path = os.path.join(args.data, f"{args.ehr}.flat.npy")

    def __getitem__(self, idx):
        # NOTE: Warning occurs when converting np.int16 read-only array into tensor, but ignoreable
        data = self.data[self.keys[idx]]["fl"]
        label = self.data[self.keys[idx]].attrs[self.pred_target]
        return {
            "input_ids": torch.IntTensor(data[0, :]),
            "type_ids": torch.IntTensor(data[1, :]),
            "dpe_ids": torch.IntTensor(data[2, :]),
            "label": label,
        }


def main():
    args = get_parser().parse_args()
    dataset = HierarchicalEHRDataset(args, "train")
    print(dataset.__getitem__(1))
    dataset = FlattenEHRDataset(args, "train")
    print(dataset.__getitem__(1))
    pass


if __name__ == "__main__":
    main()
