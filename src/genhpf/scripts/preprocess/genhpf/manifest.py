import argparse
import os
import random

import h5pickle
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR", type=str, help="root directory containing the data.h5 and label.csv"
    )
    parser.add_argument("--dest", default=".", type=str, metavar="DIR", help="output directory")
    parser.add_argument("--prefix", default="", type=str, help="prefix for the output files")
    parser.add_argument(
        "--valid-percent",
        default=0.1,
        type=float,
        metavar="D",
        help="percentage of data to use as validation and test set (between 0 and 0.5)",
    )
    parser.add_argument("--seed", default=42, type=int, metavar="N", help="random seed")
    return parser


def main(args):
    assert 0 <= args.valid_percent <= 0.5, (
        f"Invalid valid-percent: {args.valid_percent}. " "Must be between 0 and 0.5."
    )

    if len(args.prefix) > 0 and not args.prefix.endswith("_"):
        args.prefix += "_"

    root_path = os.path.realpath(args.root)
    data_path = os.path.join(root_path, "data.h5")
    label_path = os.path.join(root_path, "label.csv")
    rand = random.Random(args.seed)

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    with (
        open(os.path.join(args.dest, f"{args.prefix}train.tsv"), "w") as train_f,
        open(os.path.join(args.dest, f"{args.prefix}valid.tsv"), "w") as valid_f,
        open(os.path.join(args.dest, f"{args.prefix}test.tsv"), "w") as test_f,
    ):
        print(data_path, file=train_f)
        print(label_path, file=train_f)
        print(data_path, file=valid_f)
        print(label_path, file=valid_f)
        print(data_path, file=test_f)
        print(label_path, file=test_f)

        def write(subjects, dest, split=None):
            for subject in tqdm(subjects, total=len(subjects), desc=split):
                print(subject, file=dest)

        data = h5pickle.File(data_path, "r")["ehr"]
        subjects = list(data.keys())
        rand.shuffle(subjects)

        valid_len = int(len(subjects) * args.valid_percent)
        test_len = int(len(subjects) * args.valid_percent)
        train_len = len(subjects) - valid_len - test_len

        train = subjects[:train_len]
        valid = subjects[train_len : train_len + valid_len]
        test = subjects[train_len + valid_len :]

        write(train, train_f, split=f"{args.prefix}train")
        write(valid, valid_f, split=f"{args.prefix}valid")
        write(test, test_f, split=f"{args.prefix}test")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
