import logging
from typing import Union

import numpy as np
import torch.utils.data

from genhpf.configs import ChoiceEnum

MASK_UNIT_CHOICES = ChoiceEnum(["token", "event"])

logger = logging.getLogger(__name__)


class BaseDataset(torch.utils.data.Dataset):
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def collator(self, samples):
        raise NotImplementedError

    def mask(
        self,
        tokens: Union[np.ndarray, torch.Tensor],
        mask_prob: float,
        vocab_size: int,
        mask_token_id: int,
        mask_unit: MASK_UNIT_CHOICES = "token",
        sep_token_id: int = None,
    ):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        assert 0.0 < mask_prob < 1.0, "mask_prob must be in the range (0.0, 1.0)"

        if isinstance(tokens, np.ndarray):
            tokens = torch.LongTensor(tokens)
        tokens = tokens.long()
        labels = tokens.clone()

        assert tokens.dim() == 2, (
            "input tokens must be 2D tensor, where the first dimension is the number of embeddings "
            "(i.e., input_ids, type_ids, dpe_ids) and the second dimension is the length of token "
            "sequence."
        )

        if mask_unit == "token":
            probability_matrix = torch.full((labels.size(-1),), mask_prob)
            # do not mask special tokens
            if not hasattr(self, "tokenizer"):
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

            special_tokens_mask = torch.tensor(
                self.tokenizer.get_special_tokens_mask(labels[0], already_has_special_tokens=True),
                dtype=torch.bool,
            )
            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            probability_matrix[torch.where(labels[0] == self.pad_token_id)] = 0.0

            mask_indices = torch.bernoulli(probability_matrix).bool()
            while mask_indices.sum() == 0:
                mask_indices = torch.bernoulli(probability_matrix).bool()
        elif mask_unit == "event":
            if sep_token_id is None:
                logger.warning(
                    "sep_token_id is not provided. Using the default [SEP] token id (102) as "
                    "the event delimiter."
                )
                sep_token_id = 102  # token id for [SEP]
            event_indices = torch.where(tokens[0] == sep_token_id)[0]
            assert len(event_indices) > 1, (
                "there must be at least two events in the input sequence to apply span masking. "
                "check if you are using the hierarchical structure which is not supporting the "
                "span masking method."
            )
            mask_indices = torch.zeros_like(tokens[0]).bool()
            masked_event_indices = torch.randperm(len(event_indices) - 1)[
                : round((len(event_indices) - 1) * mask_prob)
            ]
            for i in masked_event_indices:
                mask_indices[event_indices[i] : event_indices[i + 1]] = True
        else:
            raise ValueError(f"mask_unit must be one of {MASK_UNIT_CHOICES}")

        labels[:, ~mask_indices] = self.ignore_index  # we only compute loss on masked tokens

        # 80% of the time, we replace the masked input tokens with tokenizer.mask_token ([MASK])
        mask_positions = torch.bernoulli(torch.full(mask_indices.shape, 0.8)).bool() & mask_indices
        tokens[0, mask_positions] = mask_token_id  # for input_ids
        tokens[1, mask_positions] = 4  # for type_ids
        tokens[2, mask_positions] = 15  # for dpe_ids

        # 10% of the time, we replace the masked input tokens with random word
        random_positions = (
            torch.bernoulli(torch.full(mask_indices.shape, 0.5)).bool() & mask_indices & ~mask_positions
        )
        random_words = torch.randint(vocab_size, mask_indices.shape, dtype=torch.long)
        tokens[0, random_positions] = random_words[random_positions]
        random_types = torch.randint(7, mask_indices.shape, dtype=torch.long)
        tokens[1, random_positions] = random_types[random_positions]
        random_dpes = torch.randint(16, mask_indices.shape, dtype=torch.long)
        tokens[2, random_positions] = random_dpes[random_positions]

        # the rest of the time, we keep the masked input tokens unchanged
        return tokens, labels
