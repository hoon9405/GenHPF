import torch
import torch.nn as nn
import math

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models import register_model

from models.utils import PositionalEncoding, get_slopes

@register_model("eventaggregator")
class EventAggregator(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args

        self.pos_encoder = None
       
        if args.time_embed =='sinusoidal':    
            self.pos_encoder = PositionalEncoding(
                args.pred_dim, args.dropout, args.max_seq_len
            )

        elif args.time_embed=='alibi_time':
            self.register_buffer(
            "slopes",
            torch.Tensor(get_slopes(args.n_heads, args.alibi_const)),
            )


        self.layer_norm = nn.LayerNorm(args.embed_dim, eps=1e-12)

        encoder_layers = TransformerEncoderLayer(
            args.pred_dim,
            args.n_heads,   # default=8
            args.pred_dim*4,
            args.dropout,
            batch_first=True
        )
        
        self.transformer_encoder = TransformerEncoder(encoder_layers, args.n_layers)

    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)

    def forward(self, x, input_ids, times, **kwargs):
        # input_ids: (B, S) (B x S, W ) -> (Bx s, W) -> (B, s, W)
        
        B, S = input_ids.shape[0], input_ids.shape[1]
        if self.args.structure=='hi': 
            # True for padded events (B, S)
            src_pad_mask = input_ids[:, :, 1].eq(0).to(x.device)
        else:
            src_pad_mask = input_ids.eq(0).to(x.device) 
  
        x, src_mask = self.forward_transformer_pos_enc(x, times)

        x = self.layer_norm(x)
        # x: (B, S, E)
        # For each event, attend to all other non-pad events in the same icustay
        encoder_output = self.transformer_encoder(x, mask=src_mask, src_key_padding_mask=src_pad_mask)

        return encoder_output

    def forward_transformer_pos_enc(self, x, times):
        # NOTE: ALiBi Mask should be the order of (B*n_heads, S, S)
        B, S = times.shape
        if self.args.time_embed == "sinusoidal":
            x = self.pos_encoder(x)
            mask = None
            
        elif self.args.time_embed == "alibi_time":
            mask = (
                torch.einsum("i, jk -> jik", self.slopes, times)
                .reshape(-1, 1, S)
                .repeat(1, S, 1)
            ).to(torch.float32)
        
        return x, mask