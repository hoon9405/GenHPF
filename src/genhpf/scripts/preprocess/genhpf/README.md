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

### Arguments Description
- `--dest`: Output directory path (e.g., `--dest $HOME/output/mimiciv/`)
- `--ehr`: Type of EHR dataset (eicu/mimiciii/mimiciv)
- `--data`: Path to the input data directory (e.g., `--data $HOME/data/mimiciv/2.2/`)
- `--first_icu`: Process only first ICU admission
- `--emb_type`: Embedding type ("textbase" or "codebase")
- `--feature`: Feature selection mode ("all_features" or "select")

### Prediction Tasks
You can add any combination of the following prediction tasks:
- `--readmission`
- `--mortality`
- `--los_3day`
- `--los_7day`
- `--long_term_mortality`
- `--final_acuity`
- `--imminent_discharge`
- `--diagnosis`
- `--creatinine`
- `--bilirubin`
- `--platelets`
- `--wbc`

### Baseline Model Requirements

To reproduce the experiments from the GenHPF paper (including other baseline models), use the following configurations for data preprocessing:
- **GenHPF**: `--emb_type textbase --feature all_features`
- **SAND**: `--emb_type codebase --feature select`
- **DescEmb**: `--emb_type textbase --feature select`
- **Rajkomar**: `--emb_type textbase --feature all_features`

Example Command for GenHPF model data preparation:
```shell script
python3 main.py \
    --dest $HOME/output/mimiciv/ \
    --ehr mimiciv \
    --data $HOME/data/mimiciv/2.2/ \
    --emb_type textbase \
    --feature all_features \
    --first_icu \
    --mortality --readmission  # add desired prediction tasks
```

### Cache Option
- `--cache`: Enable caching of intermediate processing results
  - When enabled, the pipeline reuses previously processed data from cache instead of reprocessing
  - Cache files are stored in `~/.cache/ehr` directory
  - **Note**: Cache will use the previously processed task labels (mortality, readmission, etc.) from the cached results.

### Resource Requirements
- Full pipeline processing (all tables) for each dataset (MIMIC-III, MIMIC-IV, eICU) requires:
  - ~180GB RAM
  - ~6 hours on 128 cores (AMD EPYC 7502 32-Core Processor)

### External Resources
If automatic download fails, manually download these files and place them in the cache directory:
- CCS Diagnosis Codes: [ccs_multi_dx_tool_2015.csv](https://www.hcup-us.ahrq.gov/toolssoftware/ccs/Multi_Level_CCS_2015.zip)
- ICD Conversion: [icd10cmtoicd9gem.csv](https://data.nber.org/gem/icd10cmtoicd9gem.csv)
