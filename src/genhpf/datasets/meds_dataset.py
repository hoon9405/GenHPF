import logging
import os
from typing import List, Union

import h5pickle
import numpy as np
import torch

from genhpf.datasets.dataset import BaseDataset

logger = logging.getLogger(__name__)


class MEDSDataset(BaseDataset):
    def __init__(
        self,
        manifest_paths: List[str],
        structure: str = "hierarchical",
        vocab_size: int = 28996,
        pad_token_id: int = 0,
        sep_token_id: int = 102,
        ignore_index: int = -100,
        apply_mask: bool = False,
        mask_token_id: int = 103,
        mask_prob: float = 0,
        mask_unit: str = "individual",
        simclr: bool = False,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__()

        if structure == "hierarchical":
            structure = "hi"
        elif structure == "flattened":
            raise NotImplementedError("Flattened structure is not supported yet.")
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
        self.shard_ids = []
        self.sizes = []
        for i, manifest_path in enumerate(manifest_paths):
            with open(manifest_path, "r") as f:
                data_i_root = f.readline().strip()
                shard_ids = []
                for j, line in enumerate(f):
                    if debug and j >= 300:
                        break
                    items = line.strip().split("\t")
                    assert len(items) == 3, line
                    subject_id, num_events, shard_id = items
                    self.subjects.append((i, subject_id))
                    self.sizes.append(int(num_events))
                    shard_ids.append(int(shard_id))

                data_i = {}
                unique_shard_ids = np.unique(shard_ids)
                for shard_id in unique_shard_ids:
                    data_i[shard_id] = h5pickle.File(os.path.join(data_i_root, f"{shard_id}.h5"))["ehr"]
                self.data.append(data_i)
                self.shard_ids.extend(shard_ids)
        logger.info(f"loaded {len(self.subjects)} samples from {len(manifest_paths)} dataset(s)")

    def __len__(self):
        return len(self.subjects)


class HierarchicalMEDSDataset(MEDSDataset):
    def __init__(
        self,
        manifest_paths: List[str],
        max_events: int = 256,
        label: bool = False,
        tasks: List[str] = None,
        num_labels: List[int] = None,
        dummy_token_id: int = 101,
        **kwargs,
    ):
        kwargs.pop("structure", None)
        super().__init__(manifest_paths=manifest_paths, structure="hierarchical", **kwargs)

        self.max_events = max_events
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
        target_size = self.max_events

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
                collated_input_ids[i] = input_ids[i][-target_size:]
                collated_type_ids[i] = type_ids[i][-target_size:]
                collated_dpe_ids[i] = dpe_ids[i][-target_size:]

        out = {"id": [s["id"] for s in samples]}
        out["net_input"] = {
            "input_ids": collated_input_ids,
            "type_ids": collated_type_ids,
            "dpe_ids": collated_dpe_ids,
        }

        if self.label:
            label = {}
            for task in self.tasks:
                if len(samples[0][task]) == 1:
                    label[task] = torch.cat([s[task] for s in samples])
                else:
                    label[task] = torch.stack([s[task] for s in samples])
            out["label"] = label

        return out

    def __getitem__(self, idx):
        data_idx, subject = self.subjects[idx]
        data = self.data[data_idx][self.shard_ids[idx]][subject]

        tokens = data[self.structure][:]
        if self.apply_mask:
            tokens = self.mask(
                tokens,
                mask_prob=self.mask_prob,
                vocab_size=self.vocab_size,
                mask_token_id=self.mask_token_id,
                mask_unit=self.mask_unit,
                sep_token_id=self.sep_token_id,
            )

        ret = {
            "id": subject,
            "input_ids": torch.LongTensor(tokens[:, 0, :]),
            "type_ids": torch.LongTensor(tokens[:, 1, :]),
            "dpe_ids": torch.LongTensor(tokens[:, 2, :]),
        }

        if self.label:
            for i, task in enumerate(self.tasks):
                try:
                    ret[task] = torch.LongTensor(data["label"][i])
                except ValueError:
                    ret[task] = torch.LongTensor([data["label"][()]])
        return ret
