import logging

import torch
import torch.nn as nn

from modules import GatherLayer

from models import register_model, MODEL_REGISTRY

logger = logging.getLogger(__name__)

@register_model("GenHPF_simclr")
class GenHPF_SimCLR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.emb_type_model = self._emb_type_model.build_model(args)
        self.pred_model = self._pred_model.build_model(args)
        
    @property
    def _emb_type_model(self):
        if self.args.structure=='hi':
            return MODEL_REGISTRY['eventencoder']
        else:
            return MODEL_REGISTRY['FeatEmb']
    
    @property
    def _pred_model(self):
        return MODEL_REGISTRY[self.args.pred_model]
    
    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)

    @classmethod
    def from_pretrained(cls, args, checkpoint=None, state_dict=None):
        model = cls(args)

        if state_dict is None: 
            state_dict = torch.load(checkpoint, map_location='cpu')['model']     
        
        #Transfer learning codebase emb
        if args.train_task =='finetune' and args.emb_type=='codebase':
            state_dict = {
                    k: v for k,v in state_dict.items() if (
                        ('input2emb' not in k) and ('pos_enc' not in k)
                    )
                }
        
        model.load_state_dict(state_dict)
    
        return model 

    def forward(self, **kwargs):

        all_embs = self.emb_type_model(**kwargs)  # (B, S, E)
        x = self.pred_model(all_embs, **kwargs)

        input_ids = kwargs['input_ids']
        B, S = input_ids.size(0),  input_ids.size(1) 

        if self.args.pred_pooling == 'cls':
            x = x[:, 0, :]
        elif self.args.pred_pooling == 'mean': #logit checked
            if self.args.structure =='hi': 
                mask = ~input_ids[:, :, 1].eq(0)
            else:
                mask = (input_ids!=0)
            mask = mask.unsqueeze(dim=2).to(x.device).expand(B, S, self.args.pred_dim)
            x = (x*mask).sum(dim=1)/mask.sum(dim=1)
            #x = x.mean(dim=1)

        # input x: [B, model_dim]            
        # GatherLayer.apply(x): List[Tensor]: [world_size, B, model]
        if self.args.world_size > 1: #TODO: check
            x = torch.cat(GatherLayer.apply(x), dim=0)

        output = {
            'logits': x,
        }

        return output


    def get_outputs(self, net_output):

        logits = net_output['logits']
        
        return logits

