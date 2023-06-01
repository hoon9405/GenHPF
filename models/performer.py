import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from performer_pytorch import Performer
from models import register_model
from models.utils import PositionalEncoding

@register_model("performer_pred")
class PerformerPred(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args

        self.pos_encoder = None
        if args.structure =='hi':
            self.pos_encoder = PositionalEncoding(
                args.pred_dim, args.dropout, args.max_seq_len
                )
            self.layer_norm = nn.LayerNorm(args.embed_dim, eps=1e-12)

        
        self.performer_encoder = Performer(
            dim = args.pred_dim,
            depth = args.n_layers,
            heads = args.n_heads,
            dim_head = 64,
            local_attn_heads = 0,
            local_window_size = 256,
            causal = False,
            ff_mult = 4,
            nb_features = 64,
            feature_redraw_interval = 1000,
            reversible = True,
            ff_chunks = 1,
            generalized_attention = True,  # turn on True
            kernel_fn = nn.ReLU(),
            use_scalenorm = False,
            use_rezero = False,
            ff_glu = False,
            ff_dropout = args.dropout,
            attn_dropout = args.dropout,
            cross_attend = False,
            no_projection = False,
            auto_check_redraw = True,
            qkv_bias = True,
            attn_out_bias = True,
            shift_tokens = True
        )

    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)

    def forward(self, x, input_ids, **kwargs):
        # input_ids: (B, S)
        B, S = input_ids.shape[0], input_ids.shape[1]
        if self.args.structure =='hi': 
            src_pad_mask = input_ids[:, :, 1].eq(0).to(x.device)
        else:
            src_pad_mask = input_ids.eq(0).to(x.device) 
        
        src_mask= None
        
        # if you want window sift mask, should src_mask include pad mask 
        if self.pos_encoder is not None:
            x = self.layer_norm(self.pos_encoder(x))

        encoder_output = self.performer_encoder(x, mask=src_pad_mask)

        return encoder_output