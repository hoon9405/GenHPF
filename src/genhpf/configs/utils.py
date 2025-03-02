from dataclasses import is_dataclass
from omegaconf import OmegaConf, open_dict

from genhpf.configs import BaseConfig

def merge_with_parent(dc: BaseConfig, cfg: BaseConfig, remove_missing=False):
    if remove_missing:

        def remove_missing_rec(src_keys, target_cfg):
            if is_dataclass(target_cfg):
                target_keys = set(target_cfg.__dataclass_fields__.keys())
            else:
                target_keys = set(target_cfg.keys())
            
            for k in list(src_keys.keys()):
                if k not in target_keys:
                    del src_keys[k]
                elif OmegaConf.is_config(src_keys[k]):
                    tgt = getattr(target_cfg, k)
                    if tgt is not None and (is_dataclass(tgt) or hasattr(tgt, "keys")):
                        remove_missing_rec(src_keys[k], tgt)
    
        with open_dict(cfg):
            remove_missing_rec(cfg, dc)

    merged_cfg = OmegaConf.merge(dc, cfg)
    merged_cfg.__dict__["_parent"] = cfg.__dict__["_parent"]
    OmegaConf.set_struct(merged_cfg, True)
    return merged_cfg