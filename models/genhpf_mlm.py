import logging

import torch.nn as nn

from models import register_model, MODEL_REGISTRY

logger = logging.getLogger(__name__)

@register_model("GenHPF_mlm")
class GenHPF(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.emb_type_model = self._emb_type_model.build_model(args)
        self.pred_model = self._pred_model.build_model(args)
        self.emb2out_model = self._emb2out_model.build_model(args)
        
    #TODO: group emb_type, pred_model, emb2out to parent model
    @property
    def _emb_type_model(self):
        if self.args.structure=='hi':
            return MODEL_REGISTRY['eventencoder']
        else:
            return MODEL_REGISTRY['FeatEmb']
    
    @property
    def _pred_model(self):
        return MODEL_REGISTRY[self.args.pred_model]
    
    @property
    def _emb2out_model(self):
        return MODEL_REGISTRY['mlmout']
    
    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)

    @classmethod
    def from_pretrained(cls, args, state_dict=0):
        model = cls(args)
        model.load_state_dict(state_dict)
    
        return model 

    def forward(self, **kwargs):

        all_codes_embs = self.emb_type_model(**kwargs)  # (B, S, E)
        x = self.pred_model(all_codes_embs, **kwargs)
        net_output = self.emb2out_model(x, **kwargs)

        output = {
            'logits': net_output,
        }

        return output


    def get_outputs(self, net_output, task=None, normalize=None):
        return net_output['logits']
        

    def get_targets(self, sample, net_output=None, task=None):

        return sample['net_input']
