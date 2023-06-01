import torch
import torch.nn as nn
from models import register_model

@register_model("predout")
class PredOutPutLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.final_proj = nn.ModuleDict()

        for task in self.args.pred_tasks:
            self.final_proj[task.name] = nn.Linear(
                args.pred_dim, task.num_classes
            )
   
    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)


    def forward(self, x, input_ids, **kwargs):
        B, S = input_ids.size(0),  input_ids.size(1) 
        if self.args.pred_pooling =='cls':
            x = x[:, 0, :]
        elif self.args.pred_pooling =='mean':
            if self.args.structure =='hi': 
                mask = ~input_ids[:, :, 1].eq(102)
            else:
                mask = (input_ids!=0)
            mask = mask.unsqueeze(dim=2).to(x.device).expand(B, S, self.args.pred_dim)
            x = (x*mask).sum(dim=1)/mask.sum(dim=1)
            #x = x.mean(dim=1)
        
        preds = dict()

        for k, layer in self.final_proj.items():
            preds[k] = layer(x)

        return {'pred_output': preds}


@register_model("mlmout")
class MLMOutPutLayer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        input_index_size = 28996
        type_index_size = 7
        dpe_index_size = 16 # for pooled
        
        self.input_ids_out = nn.Linear(
            args.embed_dim,
            input_index_size
        ) 
        
        self.type_ids_out = nn.Linear(
            args.embed_dim,
            type_index_size 
        ) if 'type' in self.args.mask_list else None

        self.dpe_ids_out = nn.Linear(
            args.embed_dim,
            dpe_index_size
        ) if 'dpe' in self.args.mask_list else None
    
    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)

    def forward(self, x, **kwargs):
        input_ids = self.input_ids_out(x)
        type_ids = self.type_ids_out(x)  if self.type_ids_out else None
        dpe_ids = self.dpe_ids_out(x) if self.dpe_ids_out else None

        return {
            'input_ids' : input_ids,
            'type_ids' : type_ids,
            'dpe_ids' : dpe_ids
        }
