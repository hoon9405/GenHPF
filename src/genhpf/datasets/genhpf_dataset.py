import logging
from typing import List, Union

import h5pickle
import numpy as np
import pandas as pd
import torch

from genhpf.datasets.dataset import BaseDataset

logger = logging.getLogger(__name__)


class GenHPFDataset(BaseDataset):
    def __init__(
        self,
        manifest_paths: List[str],
        structure: str,
        vocab_size: int = 28996,
        pad_token_id: int = 0,
        sep_token_id: int = 102,
        ignore_index: int = -100,
        apply_mask: bool = False,
        mask_token_id: int = 103,
        mask_prob: float = 0,
        mask_unit: str = "individual",
        simclr: bool = False,
        **kwargs,
    ):
        super().__init__()

        if structure == "hierarchical":
            structure = "hi"
        elif structure == "flattened":
            structure = "fl"
        self.structure = structure

        self.pad_token_id = pad_token_id

        self.ignore_index = ignore_index

        self.apply_mask = apply_mask
        self.mask_prob = mask_prob
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.mask_unit = mask_unit
        self.sep_token_id = sep_token_id

        self.simclr = simclr

        for k, v in kwargs.items():
            self.__setattr__(k, v)

        self.data = []
        self.subjects = []
        self.labels = {}
        for i, manifest_path in enumerate(manifest_paths):
            with open(manifest_path, "r") as f:
                data_root = f.readline().strip()
                label_root = f.readline().strip()
                self.data.append(h5pickle.File(data_root, "r")["ehr"])
                labels = pd.read_csv(label_root)
                labels.index = [(i, str(x)) for x in labels["stay_id"]]
                labels = labels.drop(columns=["stay_id"])
                self.labels.update(labels.to_dict(orient="index"))
                for line in f:
                    items = line.strip().split("\t")
                    assert len(items) == 1, line
                    self.subjects.append((i, items[0]))
        logger.info(f"loaded {len(self.subjects)} samples from {len(manifest_paths)} dataset(s)")

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, index):
        raise NotImplementedError


class HierarchicalGenHPFDataset(GenHPFDataset):
    def __init__(
        self,
        manifest_paths: List[str],
        label: bool = False,
        tasks: List[str] = None,
        num_labels: List[int] = None,
        dummy_token_id: int = 101,
        **kwargs,
    ):
        kwargs.pop("structure", None)
        super().__init__(manifest_paths=manifest_paths, structure="hierarchical", **kwargs)

        self.label = label
        self.tasks = tasks
        self.num_labels = num_labels
        self.dummy_token_id = dummy_token_id

    def mask(self, tokens: Union[np.ndarray, torch.Tensor], **kwargs):
        for i, event in enumerate(tokens):
            tokens[i], _ = super().mask(event, **kwargs)
        return tokens

    def collator(self, samples):
        samples = [s for s in samples if s["input_ids"] is not None]
        if len(samples) == 0:
            return {}

        if self.simclr:
            input_ids = sum(
                [
                    [s["input_ids"][: len(s["input_ids"]) // 2], s["input_ids"][len(s["input_ids"]) // 2 :]]
                    for s in samples
                ],
                [],
            )
            type_ids = sum(
                [
                    [s["type_ids"][: len(s["input_ids"]) // 2], s["type_ids"][len(s["input_ids"]) // 2 :]]
                    for s in samples
                ],
                [],
            )
            dpe_ids = sum(
                [
                    [s["dpe_ids"][: len(s["input_ids"]) // 2], s["dpe_ids"][len(s["input_ids"]) // 2 :]]
                    for s in samples
                ],
                [],
            )
        else:
            input_ids = [s["input_ids"] for s in samples]
            type_ids = [s["type_ids"] for s in samples]
            dpe_ids = [s["dpe_ids"] for s in samples]

        sizes = [s.size(0) for s in input_ids]
        target_size = max(sizes)

        collated_input_ids = (
            input_ids[0].new_zeros((len(input_ids), target_size, len(input_ids[0][0]))).long()
        )
        collated_type_ids = type_ids[0].new_zeros((len(type_ids), target_size, len(type_ids[0][0]))).long()
        collated_dpe_ids = dpe_ids[0].new_zeros((len(dpe_ids), target_size, len(dpe_ids[0][0]))).long()
        for i, size in enumerate(sizes):
            diff = size - target_size
            if diff == 0:
                collated_input_ids[i] = input_ids[i]
                collated_type_ids[i] = type_ids[i]
                collated_dpe_ids[i] = dpe_ids[i]
            elif diff < 0:
                collated_input_ids[i] = torch.cat(
                    [
                        input_ids[i],
                        input_ids[i].new_zeros(-diff, len(input_ids[i][0])),
                    ],
                    dim=0,
                )
                # add dummy token to the start of each padded event as the event encoder can be
                # crushed when all the input tokens are pad tokens
                collated_input_ids[i][diff:, 0] = self.dummy_token_id
                collated_type_ids[i] = torch.cat(
                    [
                        type_ids[i],
                        type_ids[i].new_zeros(-diff, len(type_ids[i][0])),
                    ],
                    dim=0,
                )
                collated_dpe_ids[i] = torch.cat(
                    [
                        dpe_ids[i],
                        dpe_ids[i].new_zeros(-diff, len(dpe_ids[i][0])),
                    ],
                    dim=0,
                )
            else:
                raise ValueError(f"size mismatch, expected <={target_size}, got {size}")

        out = {"id": [s["id"] for s in samples]}
        out["net_input"] = {
            "input_ids": collated_input_ids,
            "type_ids": collated_type_ids,
            "dpe_ids": collated_dpe_ids,
        }

        if self.label:
            label = {}
            for task in self.tasks:
                label[task] = torch.stack([s[task] for s in samples])
            out["label"] = label

        return out

    def __getitem__(self, index):
        data_index, subject = self.subjects[index]
        data = self.data[data_index][subject][self.structure][:]

        if self.apply_mask:
            data = self.mask(
                data,
                mask_prob=self.mask_prob,
                vocab_size=self.vocab_size,
                mask_token_id=self.mask_token_id,
                mask_unit=self.mask_unit,
                sep_token_id=self.sep_token_id,
            )

        ret = {
            "id": subject,
            "input_ids": torch.LongTensor(data[:, 0, :]),
            "type_ids": torch.LongTensor(data[:, 1, :]),
            "dpe_ids": torch.LongTensor(data[:, 2, :]),
        }

        if self.label:
            for i, task in enumerate(self.tasks):
                ret[task] = self.labels[self.subjects[index]][task]
                if isinstance(ret[task], str):
                    ret[task] = eval(ret[task])
                # for multi-label classification, where the label is given by a list of class indices
                if isinstance(ret[task], list):
                    ret[task] = list(map(int, ret[task]))
                    num_label = self.num_labels[i]
                    label = np.zeros(num_label, dtype=np.int16)
                    label[ret[task]] = 1
                    ret[task] = torch.tensor(label)
                else:
                    if np.isnan(ret[task]) or ret[task] < 0:
                        ret[task] = self.ignore_index
                    ret[task] = torch.tensor(ret[task])
        return ret


class FlattenedGenHPFDataset(GenHPFDataset):
    def __init__(
        self,
        manifest_paths: List[str],
        label: bool = False,
        tasks: List[str] = None,
        num_labels: List[int] = None,
        **kwargs,
    ):
        kwargs.pop("structure", None)
        super().__init__(manifest_paths=manifest_paths, structure="flattened", **kwargs)

        self.label = label
        self.tasks = tasks
        self.num_labels = num_labels

    def sample_crop_indices(self, size, diff):
        if self.mask:
            start = np.random.randint(0, diff + 1)
            end = size - diff + start
        else:
            start = 0
            end = size - diff
        return start, end

    def pad_to_max_size(self, sample, max_len):
        if len(sample) < max_len:
            sample = np.concatenate([sample, np.zeros(max_len - len(sample), dtype=np.int16)])
        else:
            sample = sample[:max_len]
        return sample

    def collator(self, samples):
        samples = [s for s in samples if s["input_ids"] is not None]
        if len(samples) == 0:
            return {}

        if self.simclr:
            input_ids = sum(
                [
                    [s["input_ids"][: len(s["input_ids"]) // 2], s["input_ids"][len(s["input_ids"]) // 2 :]]
                    for s in samples
                ],
                [],
            )
            type_ids = sum(
                [
                    [s["type_ids"][: len(s["input_ids"]) // 2], s["type_ids"][len(s["input_ids"]) // 2 :]]
                    for s in samples
                ],
                [],
            )
            dpe_ids = sum(
                [
                    [s["dpe_ids"][: len(s["input_ids"]) // 2], s["dpe_ids"][len(s["input_ids"]) // 2 :]]
                    for s in samples
                ],
                [],
            )
            if self.apply_mask:
                input_label = sum(
                    [
                        [
                            s["input_label"][: len(s["input_ids"]) // 2],
                            s["input_label"][len(s["input_ids"]) // 2 :],
                        ]
                        for s in samples
                    ],
                    [],
                )
        else:
            input_ids = [s["input_ids"] for s in samples]
            type_ids = [s["type_ids"] for s in samples]
            dpe_ids = [s["dpe_ids"] for s in samples]
            if self.apply_mask:
                input_label = [s["input_label"] for s in samples]
                type_label = [s["type_label"] for s in samples]
                dpe_label = [s["dpe_label"] for s in samples]

        sizes = [s.size(0) for s in input_ids]
        target_size = max(sizes)

        collated_input_ids = input_ids[0].new_zeros((len(input_ids), target_size)).long()
        collated_type_ids = type_ids[0].new_zeros((len(type_ids), target_size)).long()
        collated_dpe_ids = dpe_ids[0].new_zeros((len(dpe_ids), target_size)).long()
        if self.apply_mask:
            collated_input_label = input_label[0].new_zeros((len(input_label), target_size)).long()
            collated_type_label = type_label[0].new_zeros((len(type_label), target_size)).long()
            collated_dpe_label = dpe_label[0].new_zeros((len(dpe_label), target_size)).long()
        for i, size in enumerate(sizes):
            diff = size - target_size
            if diff == 0:
                collated_input_ids[i] = input_ids[i]
                collated_type_ids[i] = type_ids[i]
                collated_dpe_ids[i] = dpe_ids[i]
                if self.apply_mask:
                    collated_input_label[i] = input_label[i]
                    collated_type_label[i] = type_label[i]
                    collated_dpe_label[i] = dpe_label[i]
            elif diff < 0:
                collated_input_ids[i] = torch.cat(
                    [
                        input_ids[i],
                        input_ids[i].new_zeros(
                            -diff,
                        ),
                    ],
                    dim=0,
                )
                collated_type_ids[i] = torch.cat(
                    [
                        type_ids[i],
                        type_ids[i].new_zeros(
                            -diff,
                        ),
                    ],
                    dim=0,
                )
                collated_dpe_ids[i] = torch.cat(
                    [
                        dpe_ids[i],
                        dpe_ids[i].new_zeros(
                            -diff,
                        ),
                    ],
                    dim=0,
                )
                if self.apply_mask:
                    collated_input_label[i] = torch.cat(
                        [
                            input_label[i],
                            input_label[i].new_zeros(
                                -diff,
                            ),
                        ],
                        dim=0,
                    )
                    collated_type_label[i] = torch.cat(
                        [
                            type_label[i],
                            type_label[i].new_zeros(
                                -diff,
                            ),
                        ],
                        dim=0,
                    )
                    collated_dpe_label[i] = torch.cat(
                        [
                            dpe_label[i],
                            dpe_label[i].new_zeros(
                                -diff,
                            ),
                        ],
                        dim=0,
                    )
            else:
                raise ValueError(f"size mismatch, expected <={target_size}, got {size}")

        out = {"id": [s["id"] for s in samples]}
        out["net_input"] = {
            "input_ids": collated_input_ids,
            "type_ids": collated_type_ids,
            "dpe_ids": collated_dpe_ids,
        }

        if self.apply_mask:
            out["input_label"] = collated_input_label
            out["type_label"] = collated_type_label
            out["dpe_label"] = collated_dpe_label

        if self.label:
            label = {}
            for task in self.tasks:
                label[task] = torch.stack([s[task] for s in samples])
            out["label"] = label

        return out

    def __getitem__(self, index):
        data_index, subject = self.subjects[index]
        data = self.data[data_index][subject][self.structure][:]

        if self.apply_mask:
            data, mlm_labels = self.mask(
                data,
                mask_prob=self.mask_prob,
                vocab_size=self.vocab_size,
                mask_token_id=self.mask_token_id,
                mask_unit=self.mask_unit,
                sep_token_id=self.sep_token_id,
            )

        ret = {
            "id": self.subjects[index],
            "input_ids": torch.LongTensor(data[0, :]),
            "type_ids": torch.LongTensor(data[1, :]),
            "dpe_ids": torch.LongTensor(data[2, :]),
        }
        if self.apply_mask:
            ret["input_label"] = torch.LongTensor(mlm_labels[0, :])
            ret["type_label"] = torch.LongTensor(mlm_labels[1, :])
            ret["dpe_label"] = torch.LongTensor(mlm_labels[2, :])

        if self.label:
            for i, task in enumerate(self.tasks):
                ret[task] = self.labels[self.subjects[index]][task]
                if isinstance(ret[task], str):
                    ret[task] = eval(ret[task])
                # for multi-label classification, where the label is given by a list of class indices
                if isinstance(ret[task], list):
                    ret[task] = list(map(int, ret[task]))
                    num_label = self.num_labels[i]
                    label = np.zeros(num_label, dtype=np.int16)
                    label[ret[task]] = 1
                    ret[task] = torch.tensor(label)
                else:
                    if np.isnan(ret[task]) or ret[task] < 0:
                        ret[task] = self.ignore_index
                    ret[task] = torch.tensor(ret[task])

        return ret
