from dataclasses import dataclass, field
from omegaconf import II
import logging

import torch
import torch.nn as nn

from genhpf.modules import PositionalEncoding, get_slopes
from genhpf.configs import BaseConfig, ChoiceEnum

logger = logging.getLogger(__name__)

GENHPF_MODEL_ARCH_CHOICES = ChoiceEnum(["hierarchical", "flattened"])
GENHPF_AGGREGATOR_ARCH_CHOICES = ChoiceEnum(["transformer", "performer"])
GENHPF_EMBEDDING_METHOD_CHOICES = ChoiceEnum(["code", "text"])
GENHPF_TIME_EMBEDDING_METHOD_CHOICES = ChoiceEnum(["sinusoidal", "alibi"])

@dataclass
class GenHPFConfig(BaseConfig):
    structure: GENHPF_MODEL_ARCH_CHOICES = field(
        default="hierarchical",
        metadata={"help": "Architecture choice for GenHPF. Choose from hierarchical or flattened"}
    )
    embedding_method: GENHPF_EMBEDDING_METHOD_CHOICES = field(
        default="text",
        metadata={"help": "Embedding method choice for GenHPF. Choose from code or text"}
    )
    time_embedding_method: GENHPF_TIME_EMBEDDING_METHOD_CHOICES = field(
        default="sinusoidal",
        metadata={
            "help": "Time embedding method choice for the hierarchical GenHPF. Choose from "
            "sinusoidal or alibi"
        }
    )

    encoder_max_seq_len: int = field(
        default=192,
        metadata={
            "help": "max sequence length for the event encoder, only used when structure is "
            "hierarchical. this is the max number of tokens in an event."
        }
    )
    agg_max_seq_len: int = field(
        default=256,
        metadata={
            "help": "max sequence length for the event aggregator. In the hierarchical structure, "
            "this is the max number of events in a sample. In the flattened structure, this is the"
            "max sequence length of the flattened input."
        }
    )

    # configs for event encoder in hierarchical structure
    encoder_layers: int = field(
        default=2, metadata={"help": "num encoder layers in the transformer"}
    )
    encoder_embed_dim: int = field(
        default=128, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=512, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=4, metadata={"help": "num attention heads in the transformer"}
    )

    # configs for event aggregator
    agg_arch: GENHPF_AGGREGATOR_ARCH_CHOICES = field(
        default="transformer",
        metadata={
            "help": "Architecture choice for the event aggregator. Choose from transformer or "
            "performer"
        }
    )
    agg_layers: int = field(
        default=2, metadata={"help": "num layers in the transformer"}
    )
    agg_embed_dim: int = field(
        default=128, metadata={"help": "hidden dimension for the event aggregator"}
    )
    agg_ffn_embed_dim: int = field(
        default=512, metadata={"help": "hidden dimension for the FFN in the event aggregator"}
    )
    agg_attention_heads: int = field(
        default=4, metadata={"help": "num attention heads for the event aggregator"}
    )

    dropout: float = field(
        default=0.2, metadata={"help": "dropout probability"}
    )

    vocab_size: int = II("dataset.vocab_size")

class GenHPF(nn.Module):
    def __init__(self, cfg: GenHPFConfig):
        super().__init__()
        self.cfg = cfg

        self.structure = cfg.structure
        assert self.structure in GENHPF_MODEL_ARCH_CHOICES

        self.embedding_method = cfg.embedding_method
        assert self.embedding_method in GENHPF_EMBEDDING_METHOD_CHOICES

        self.time_embedding_method = cfg.time_embedding_method
        assert self.time_embedding_method in GENHPF_TIME_EMBEDDING_METHOD_CHOICES

        self.word_embeddings = nn.Embedding(cfg.vocab_size, cfg.encoder_embed_dim, padding_idx=0)
        if self.embedding_method == "text":
            # we currently use 7 token types and 16 digit places in this version
            self.token_type_vocab_size = 7
            self.digit_place_vocab_size = 16
            self.token_type_embeddings = nn.Embedding(
                self.token_type_vocab_size, cfg.encoder_embed_dim, padding_idx=0
            )
            self.digit_place_embeddings = nn.Embedding(
                self.digit_place_vocab_size, cfg.encoder_embed_dim, padding_idx=0
            )

        max_length = (
            cfg.encoder_max_seq_len if self.structure == "hierarchical" else cfg.agg_max_seq_len
        )
        self.positional_encodings = PositionalEncoding(cfg.encoder_embed_dim, cfg.dropout, max_length)

        self.event_encoder = None
        if self.structure == "hierarchical" and self.embedding_method == "text":
            self.encoder_layer_norm = nn.LayerNorm(cfg.encoder_embed_dim, eps=1e-12)
            encoder_layer = nn.TransformerEncoderLayer(
                cfg.encoder_embed_dim,
                cfg.encoder_attention_heads,
                cfg.encoder_ffn_embed_dim,
                cfg.dropout,
                batch_first=True
            )
            self.event_encoder = nn.TransformerEncoder(encoder_layer, self.cfg.encoder_layers)
            self.post_encode_proj = nn.Linear(cfg.encoder_embed_dim, cfg.agg_embed_dim)
            if self.time_embedding_method == "sinusoidal":
                self.event_positional_encodings = PositionalEncoding(
                    cfg.agg_embed_dim, cfg.dropout, cfg.agg_max_seq_len
                )
            else:
                self.register_buffer(
                    "slopes", torch.Tensor(get_slopes(cfg.encoder_attention_heads, 3))
                )

        self.agg_layer_norm = nn.LayerNorm(cfg.agg_embed_dim, eps=1e-12)
        if cfg.agg_arch == "transformer":
            agg_layer = nn.TransformerEncoderLayer(
                cfg.agg_embed_dim,
                cfg.agg_attention_heads,
                cfg.agg_ffn_embed_dim,
                cfg.dropout,
                batch_first=True
            )
            self.event_aggregator = nn.TransformerEncoder(agg_layer, cfg.agg_layers)
        elif cfg.agg_arch == "performer":
            from performer_pytorch import Performer
            self.event_aggregator = Performer(
                dim=cfg.agg_embed_dim,
                depth=cfg.agg_layers,
                heads=cfg.agg_attention_heads,
                dim_head=64,
                nb_features=64,
                reversible=True,
                generalized_attention=True,
                ff_dropout=cfg.dropout,
                attn_dropout=cfg.dropout,
                shift_tokens=True
            )
        else:
            raise NotImplementedError(f"Unsupported event aggregator architecture: {cfg.agg_arch}")

    @classmethod
    def build_model(cls, cfg):
        """Build a new model instance."""
        return cls(cfg)

    def get_logits(self, net_output):
        """Get logits from the model output."""
        raise NotImplementedError

    def get_targets(self, sample, net_output):
        """Get targets from the sample or model output."""
        raise NotImplementedError

    def forward(
        self,
        input_ids: torch.Tensor,
        times: torch.Tensor = None,
        type_ids: torch.Tensor = None,
        dpe_ids: torch.Tensor = None,
        padding_mask: torch.Tensor = None,
        encoder_only: bool = False,
        **kwargs
    ):
        x = self.word_embeddings(input_ids)
        if self.embedding_method == "text":
            if type_ids is not None:
                x += self.token_type_embeddings(type_ids)
            if dpe_ids is not None:
                x += self.digit_place_embeddings(dpe_ids)

        agg_src_mask = None
        if self.structure == "hierarchical":
            assert input_ids.ndim == 3 # (batch, num_events, num_words)
            batch_size, num_events = input_ids.shape[0], input_ids.shape[1]

            x = x.view(batch_size * num_events, -1, self.cfg.encoder_embed_dim)
            x = self.positional_encodings(x)
            x = self.encoder_layer_norm(x)
            if padding_mask is None:
                padding_mask = input_ids.view(batch_size * num_events, -1).eq(0).to(x.device)
            # x: (batch * num_events, num_words, embed_dim)
            x = self.event_encoder(x, src_key_padding_mask=padding_mask)

            if padding_mask.any():
                x[padding_mask] = 0
            x = torch.div(x.sum(dim=1), (x != 0).sum(dim=1))
            x = self.post_encode_proj(x).view(batch_size, num_events, -1) # (batch, num_events, embed_dim)
            
            padding_mask = input_ids[:, :, 1].eq(0).to(x.device)

            if self.time_embedding_method == "sinusoidal":
                x = self.event_positional_encodings(x)
            else:
                assert times is not None
                agg_src_mask = (
                    torch.einsum("i, jk -> jik", self.slopes, times)
                    .reshape(-1, 1, num_events)
                    .repeat(1, num_events, 1)
                ).to(torch.float32)
        else:
            assert input_ids.ndim == 2 # (batch, seq_len)
            x = self.positional_encodings(x) # (batch, seq_len, embed_dim)
            if padding_mask is None:
                padding_mask = input_ids.eq(0).to(x.device)

        if encoder_only:
            return {
                "x": x,
                "padding_mask": padding_mask,
                "agg_src_mask": agg_src_mask
            }

        x = self.event_aggregator(x, mask=agg_src_mask, src_key_padding_mask=padding_mask)

        return x, padding_mask