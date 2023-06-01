import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from models import register_model
from models.utils import PositionalEncoding

import pickle as pkl
import os
import numpy as np

import math

@register_model("FeatEmb")
class FeatEmb(nn.Module):
    def __init__(self, args, embed_dim=None):
        super().__init__()
        
        self.args = args
        
        codebook_size = {
        'all_features':{
            'mimiciii': 19131,
            'eicu': 12233,
            'mimiciv': 19396,
            'mimiciii_eicu': 19131+12233,
            'eicu_mimiciv': 12232+19396,
            'mimiciii_mimiciv': 19131+19396,
            'mimiciii_eicu_mimiciv': 19131+12233+19396 
            },
        'select':{
            'mimiciii': 12711,
            'eicu': 10067,
            'mimiciv': 13568,
            'mimiciii_eicu': 12711+10067,
            'eicu_mimiciv': 10067+13568,
            'mimiciii_mimiciv': 12710+13567,
            'mimiciii_eicu_mimiciv': 12710+10066+13567   
            }
        }
        if self.args.emb_type =='textbase':
            self.input_index_size = 28996 # bio clinical bert vocab
            self.type_index_size = 7 
            self.dpe_index_size = 16
        else:

            self.input_index_size = codebook_size[args.feature][args.train_src] # bio clinical bert vocab
            self.type_index_size = 7 
            
        self.dpe = args.dpe
        self.token_type = args.type_token
        self.pos_enc = args.pos_enc

        if embed_dim:
            self.args.embed_dim = embed_dim        

        self.input_ids_embedding = nn.Embedding(
                self.input_index_size, self.args.embed_dim, padding_idx=0
        )

        self.type_ids_embedding =nn.Embedding(
                self.type_index_size, self.args.embed_dim, padding_idx=0
        ) if self.args.type_token else None

        self.dpe_ids_embedding =nn.Embedding(
            self.dpe_index_size, self.args.embed_dim
        ) if self.args.dpe and self.args.emb_type=='textbase' else None

        max_len = args.max_seq_len if args.structure =='fl' else args.max_word_len
   
        self.pos_encoder = PositionalEncoding(  
            args.embed_dim, args.dropout, max_len
            ) if self.pos_enc else None
        
        self.layer_norm = nn.LayerNorm(args.embed_dim, eps=1e-12)

    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)

    def load_pretrained_embed_weight(self, load_dict):
        for victim, weight in load_dict:
            getattr(self, victim+'_embedding').from_pretrained(weight) 

    def forward(self, input_ids, type_ids, dpe_ids, times, only_features=False, **kwargs): # TODO: add time token - 각 token 벡터(shape: 128) 에 sinosoidal k = time token 더해줌
        B, S = input_ids.shape[0], input_ids.shape[1] # time: hi - (B, S, 1), fl - (B, S, 1)

        x = self.input_ids_embedding(input_ids)
        
        if only_features:
            if self.pos_encoder:
                x = self.pos_encoder(x)
                return x

        if self.type_ids_embedding: # column description mean 
            x += self.type_ids_embedding(type_ids) 

        if self.dpe_ids_embedding:
            x += self.dpe_ids_embedding(dpe_ids)

        if self.args.time_embed == 'encoder' and self.args.structure=='hi': #TODO: flatten
            W = input_ids.shape[2]
            times = times.unsqueeze(-1).repeat(1, 1, W).unsqueeze(-1) # (B, S, W, 1)
            div_term = torch.exp(torch.arange(0, self.args.embed_dim, 2) * (-math.log(10000.0) / self.args.embed_dim)).to(x.device) # (embed_dim/2, )
            pe = torch.zeros(B, S, W, self.args.embed_dim) # (B, S, W, embed_dim)
            pe[:, :, :, 0::2] = torch.sin(times * div_term) # (B, S, W, embed_dim)
            pe[:, :, :, 1::2] = torch.cos(times * div_term)
            x = x + pe.to(x.device) 
        
        if self.args.structure=='hi':
            x = x.view(B*S, -1, self.args.embed_dim) 
            
        if self.pos_encoder:   
            x = self.pos_encoder(x) # (B, S, W, E) -> (B*S, W, E)
        x = self.layer_norm(x)
        return x

