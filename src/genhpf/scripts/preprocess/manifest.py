import argparse
import random
import os
import logging
import h5pickle
from tqdm import tqdm

logger = logging.getLogger(__name__)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data", type=str, help="path to the .h5 file containing the data"
    )
    parser.add_argument(
        "label", type=str, help="path to the .csv file containing the labels corresponded to the data"
    )
    parser.add_argument(
        "--dest", default=".", type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--prefix", default="", type=str, help="prefix for the output files"
    )
    parser.add_argument(
        "--valid-percent",
        default=0.1,
        type=float,
        metavar="D",
        help="percentage of the data to use as validation and test set (between 0 and 0.5)"
    )
    parser.add_argument("--seed", default=42, type=int, metavar="N", help="random seed")
    return parser

def main(args):
    assert 0 <= args.valid_percent <= 0.5

    data_path = os.path.realpath(args.data)
    label_path = os.path.realpath(args.label)
    prefix = args.prefix
    if len(prefix) > 0 and not prefix.endswith("_"):
        prefix += "_"
    rand = random.Random(args.seed)

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)
    
    with (
        open(os.path.join(args.dest, f"{prefix}train.tsv"), "w") as train_f,
        open(os.path.join(args.dest, f"{prefix}valid.tsv"), "w") as valid_f,
        open(os.path.join(args.dest, f"{prefix}test.tsv"), "w") as test_f
    ):
        print(data_path, file=train_f)
        print(label_path, file=train_f)
        print(data_path, file=valid_f)
        print(label_path, file=valid_f)
        print(data_path, file=test_f)
        print(label_path, file=test_f)

        def write(subject_ids, dest, split):
            for subject_id in tqdm(subject_ids, total=len(subject_ids), desc=split):
                print(subject_id, file=dest)

        data = h5pickle.File(data_path, "r")["ehr"]
        subject_ids = list(data.keys())

        rand.shuffle(subject_ids)

        valid_len = int(len(subject_ids) * args.valid_percent)
        test_len = int(len(subject_ids) * args.valid_percent)
        train_len = len(subject_ids) - valid_len - test_len

        train = subject_ids[:train_len]
        valid = subject_ids[train_len: train_len + valid_len]
        test = subject_ids[train_len + valid_len:]

        write(train, train_f, split=f"{prefix}train")
        write(valid, valid_f, split=f"{prefix}valid")
        write(test, test_f, split=f"{prefix}test")

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)