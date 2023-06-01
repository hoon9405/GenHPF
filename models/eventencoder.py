import logging

import torch
import torch.nn as nn

from models import register_model, MODEL_REGISTRY

logger = logging.getLogger(__name__)

@register_model("eventencoder")
class EventEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pred_dim = args.pred_dim

        self.enc_model = self._enc_model.build_model(args)

        encoder_layers = nn.TransformerEncoderLayer(
            args.embed_dim,
            args.n_heads,   # default=8
            args.embed_dim*4,
            args.dropout,
            batch_first=True
            )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, args.n_layers)

        self.post_encode_proj = (
            nn.Linear(args.embed_dim, args.pred_dim)
        )

    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)

    @property
    def _enc_model(self):
        return MODEL_REGISTRY['FeatEmb']
            
    def forward(self, input_ids, **kwargs):
        
        B, S, _= input_ids.size()
        
        x = self.enc_model(input_ids, **kwargs) # (B*S, W, E)
     
        # True for padded words on each event
        src_pad_mask = input_ids.view(B*S, -1).eq(0).to(x.device) # (B, S, W) -> (B*S, W)
        # (B*S, W, E) -> (B*S, W, E)
        # For each word, attend to all other non-pad words in the same event
        #TODO: src_key_padding_mask vs. src_pad_mask
        
        encoder_output = self.transformer_encoder(x, src_key_padding_mask=src_pad_mask)

        x = encoder_output
        # zero out padded words idcs
        if src_pad_mask.shape != x.shape:
            src_pad_mask = src_pad_mask[:, :x.shape[1]]
        x[src_pad_mask] = 0
        # non pad word-level mean pooling on each event (B*S, W, E) -> (B*S, E)
        x = torch.div(x.sum(dim=1), (x!=0).sum(dim=1))
        # (B*S, E) -> (B, S, E)
        net_output = self.post_encode_proj(x).view(B, -1, self.pred_dim)
        # event representation vector
        return net_output