from typing import List

from genhpf.configs import Config

from .genhpf_dataset import BaseDataset, FlattenedGenHPFDataset, HierarchicalGenHPFDataset

__all__ = [
    "BaseDataset" "HierarchicalGenHPFDataset",
    "FlattenedGenHPFDataset",
]


def load_dataset(
    data_path: str,
    subsets: List[str],
    cfg: Config,
):
    dataset_cfg = cfg.dataset
    model_cfg = cfg.model
    criterion_cfg = cfg.criterion

    manifest_paths = [f"{data_path}/{subset.strip()}.tsv" for subset in subsets]

    if dataset_cfg.data_format == "genhpf":
        if model_cfg.structure == "hierarchical":
            dataset = HierarchicalGenHPFDataset(
                manifest_paths=manifest_paths,
                label=dataset_cfg.label,
                tasks=getattr(criterion_cfg, "task_names", None),
                num_labels=getattr(criterion_cfg, "num_labels", None),
                vocab_size=dataset_cfg.vocab_size,
                pad_token_id=dataset_cfg.pad_token_id,
                sep_token_id=dataset_cfg.sep_token_id,
                ignore_index=dataset_cfg.ignore_index,
                apply_mask=dataset_cfg.apply_mask or "mlm" in model_cfg._name,
                mask_token_id=dataset_cfg.mask_token_id,
                mask_prob=dataset_cfg.mask_prob,
                mask_unit=dataset_cfg.mask_unit,
                simclr="simclr" in model_cfg._name,
                dummy_token_id=dataset_cfg.dummy_token_id,
            )
        else:
            dataset = FlattenedGenHPFDataset(
                manifest_paths=manifest_paths,
                label=dataset_cfg.label,
                tasks=getattr(criterion_cfg, "task_names", None),
                num_labels=getattr(criterion_cfg, "num_labels", None),
                vocab_size=dataset_cfg.vocab_size,
                pad_token_id=dataset_cfg.pad_token_id,
                sep_token_id=dataset_cfg.sep_token_id,
                ignore_index=dataset_cfg.ignore_index,
                apply_mask=dataset_cfg.apply_mask or "mlm" in model_cfg._name,
                mask_token_id=dataset_cfg.mask_token_id,
                mask_prob=dataset_cfg.mask_prob,
                mask_unit=dataset_cfg.mask_unit,
                simclr="simclr" in model_cfg._name,
            )
    else:
        raise NotImplementedError(f"unsupported data format: {dataset_cfg.data_format}")

    return dataset
