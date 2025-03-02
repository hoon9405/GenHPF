# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import _MISSING_TYPE
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from genhpf.configs import Config

logger = logging.getLogger(__name__)

def hydra_init(cfg_name="config") -> None:
    cs = ConfigStore.instance()
    cs.store(name=cfg_name, node=Config)

    for k in Config.__dataclass_fields__:
        v = Config.__dataclass_fields__[k].default
        if isinstance(v, _MISSING_TYPE):
            v = Config.__dataclass_fields__[k].default_factory
            if not isinstance(v, _MISSING_TYPE):
                v = v()
        try:
            cs.store(name = k, node = v)
        except BaseException:
            logger.error(f"{k} - {v}")
            raise

def add_defaults(cfg: DictConfig) -> None:
    """This function adds default values that are stored in dataclasses that hydra doesn't know about """

    from genhpf.criterions import CRITERION_DATACLASS_REGISTRY
    from genhpf.models import MODEL_DATACLASS_REGISTRY
    from genhpf.configs.utils import merge_with_parent
    from typing import Any

    OmegaConf.set_struct(cfg, False)

    for k, v in Config.__dataclass_fields__.items():
        field_cfg = cfg.get(k)
        if field_cfg is not None and v.type == Any:
            dc = None

            if isinstance(field_cfg, str):
                field_cfg = DictConfig({"_name": field_cfg})
                field_cfg.__dict__["_parent"] = field_cfg.__dict__["_parent"]

            name = field_cfg.get("_name")

            if k == "model":
                dc = MODEL_DATACLASS_REGISTRY.get(name)
            elif k == "criterion":
                dc = CRITERION_DATACLASS_REGISTRY.get(name)
            
            if dc is not None:
                cfg[k] = merge_with_parent(dc, field_cfg)