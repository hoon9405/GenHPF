# Integrated-EHR-Pipeline
- Pre-processing code refining project in GenHPF

## Install Requirements
- NOTE: This repository requires `python>=3.9` and `Java>=8`
```
pip install numpy pandas tqdm treelib transformers pyspark
```
## How to Use
```
main.py --ehr {eicu, mimiciii, mimiciv}
```
- It automatically download the corresponding dataset from physionet, but requires appropriate certification.
- You can also use the downloaded dataset with `--data {data path}` option
- You can check sample implementation of pytorch `dataset` on `sample_dataset.py`
- You can change options(emb_type and feature) for datasets of baseline models based on provided run bash file

```
-emb_type codebase --feature "select" 
```
