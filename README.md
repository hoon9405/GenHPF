# GenHPF : General Healthcare Predictive Framework for Multi-task Multi-source Learning

GenHPF is a general healthcare predictive framework, which requires no medical domain knowledge and minimal preprocessing for multiple prediction tasks.

Our framework presents a method for embedding any form of EHR systems for prediction tasks without requiring domain-knowledge-based pre-processing, such as medical code mapping and feature selection.

This repository provides official Pytorch code to implement GenHPF, a general healthcare predictive framework.

# Getting started with GenHPF
## STEP 1: Installation
```bash
$ pip install -e ./
```

## STEP 2: Prepare training data
Download raw datasets and required tools:
* [MIMIC-III](https://physionet.org/content/mimiciii/1.4/)
* [MIMIC-IV](https://physionet.org/content/mimiciv/2.0/)
* [eICU](https://physionet.org/content/eicu-crd/2.0/)

Then, run:
```bash
genhpf-preprocess \
  --data $DATA_DIR \
  --ehr {"eicu", "mimiciii", "mimiciv"} \
  --dest $OUTPUT_DIR \
  --first_icu \
  --emb_type {"textbase", "codebase"} \
  --feature {"all_features", "select"} \
  --mortality \
  --long_term_mortality \
  ... # add desired prediction tasks
```
This will output the processed data (`data.h5` and `label.csv`) into `$DATA_DIR/data/` directory.
For detailed descriptions for each argument, see [src/genhpf/scripts/preprocess/genhpf/README.md](src/genhpf/scripts/preprocess/genhpf/README.md).
<!-- Note that pre-processing takes about 6hours in 128 cores of AMD EPYC 7502 32-Core Processor, and requires 180GB of RAM. -->

Finally, you should prepare data manifest based on the preprocessed data:
```bash
genhpf-manifest $data_dir $label_dir \
  --dest=$output_dir \
  --prefix=$prefix \
  --valid_percent=$valid_percent
```
This will generate the manifest files (e.g., `$prefix_train.tsv`, `$prefix_valid.tsv`, `$prefix_test.tsv`) to `$output_dir` based on the `$data_dir`, which contains `data.h5`, and `$label_dir`, which contains `label.csv`.
The ratio among train, valid, and test splits is decided by `$valid_percent`.
Note that this is useful to handle various concepts of training and test datasets.
For instance, if you want to use multiple datasets (e.g., mimiciv and eicu) for training and evaluate the model on each of the datasets separately, you can perform it by placing the corresponding manifest files (e.g., mimiciv_train, eicu_train, mimiciv_valid, eicu_valid, mimiciv_test, eicu_test) in the same data directory and specifying the following command-line arguments: `dataset.train_subset="mimiciv_train,eicu_train" dataset.combine_train_subsets=true dataset.valid_subset="mimiciv_valid,eicu_valid" dataset.test_subset="mimiciv_test,eicu_test"`.

## STEP 3: Training a new model
We prepared example configuration files for various models and experimental setups.
For detailed configurations, please see [configs.py](src/genhpf/configs/configs.py) and each implemented source code (e.g., [genhpf.py](src/genhpf/models/genhpf.py)).

### Examples
### Train a new GenHPF model from scratch:
```bash
$ genhpf-train dataset.data=??? --config-dir ${GENHPF_DIR}/examples/train/genhpf --config-name hierarchical_scr
```
Note that you should fill in `dataset.data=???` with a path to the directory that contains the data manifest files (e.g., `train.h5`, `valid.h5`, etc.).


### Pre-train and fine-tune a new GenHPF model:
For pre-training with SimCLR:
```bash
$ genhpf-train dataset.data=??? --config-dir ${GENHPF_DIR}/examples/pretrain/simclr/genhpf --config-name hierarchical_pt
```
For fine-tuning:
```bash
$ genhpf-train dataset.data=??? model.from_pretrained=${pretrained_checkpoint.pt} --config-dir ${GENHPF_DIR}/examples/train/genhpf --config-name hierarchical_ft
```

## Citation
If you find GenHPF useful for your research and applications, please cite using this BibTeX:
```bibtex

@article{hur2023genhpf,
  title={GenHPF: General Healthcare Predictive Framework for Multi-task Multi-source Learning},
  author={Hur, Kyunghoon and Oh, Jungwoo and Kim, Junu and Kim, Jiyoun and Lee, Min Jae and Cho, Eunbyeol and Moon, Seong-Eun and Kim, Young-Hak and Atallah, Louis and Choi, Edward},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2023},
  publisher={IEEE}
}
```

# License
This repository is MIT-lincensed.
