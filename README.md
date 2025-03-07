# GenHPF : General Healthcare Predictive Framework for Multi-task Multi-source Learning

GenHPF is a general healthcare predictive framework, which requires no medical domain knowledge and minimal preprocessing for multiple prediction tasks.

Our framework presents a method for embedding any form of EHR systems for prediction tasks without requiring domain-knowledge-based pre-processing, such as medical code mapping and feature selection.

This repository provides official Pytorch code to implement GenHPF, a general healthcare predictive framework.

# Getting started with GenHPF
## STEP 1: Installation
For developing locally:
```bash
$ pip install -e ./
```

Otherwise:
```bash
$ pip install genhpf
```

## STEP 2: Prepare training data
### Preprocessing raw datasets to reproduce GenHPF paper results (GenHPF dataset)
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

### Preprocessing MEDS dataset
We also provide a script to preprocess [MEDS](https://github.com/mmcdermott/MEDS-DEV) dataset with a cohort defined by [ACES](https://github.com/justin13601/ACES) or [MEDS-DEV](https://github.com/mmcdermott/MEDS-DEV) (see Task section) to run with GenHPF.

```bash
genhpf-preprocess-meds $MEDS_DATA_DIR \
  --cohort $MEDS_LABELS_DIR \
  --metadata_dir $MEDS_METADATA_DIR \
  --output_dir $MEDS_OUTPUT_DIR \
  --workers $NUM_WORKERS
```

* `$MEDS_DATA_DIR`: a path to the data directory containing MEDS data to be processed. It can be a directory or the exact file path with the file extension (only `.csv` or `.parquet` allowed). If provided with directory, it tries to scan all `*.csv` or `*.parquet` files contained in the directory recursively. See [this](https://github.com/mmcdermott/MEDS-DEV?tab=readme-ov-file#building-a-dataset) if you want to build a new MEDS dataset based on MIMIC-III, MIMIC-IV, and eICU.
* `$MEDS_LABELS_DIR`: a path to the label directory for a given task, which must be a result of [ACES](https://github.com/justin13601/ACES) or [MEDS-DEV](https://github.com/mmcdermott/MEDS-DEV). It can be a directory or the exact file path that has the same file extension with the MEDS dataset to be processed. The file structure of this cohort directory should be the same with the provided MEDS data directory (`$MEDS_DATA_DIR`) to match each cohort to its corresponding shard data. See [this](https://github.com/mmcdermott/MEDS-DEV?tab=readme-ov-file#extracting-a-task) to extract a cohort for a specific task defined in MEDS-DEV.
* `$MEDS_METADATA_DIR`: a path to the metadata directory for the input MEDS dataset, expected to contain `codes.parquet`. This is used to retrieve descriptions for codes in MEDS events and convert each code to the retrieved description. Note that if a code has no specific description in `codes.parquet`, it will just treat that code as a plain text and process the event as it is.
* `$MEDS_OUTPUT_DIR`: directory to save processed outputs.
  * Enabling `--rebase` will renew this directory.
* `$NUM_WORKERS`: number of parallel workers to multi-process the script.
* **NOTE: if you encounter this error: _"polars' maximum length reached. consider installing 'polars-u64-idx'"_, please consider using more workers or installing polars-u64-idx by `pip install polars-u64-idx`.**

As a result, you will have `.h5` and `.tsv` files that has a following respective structure:
* `*.h5`
  ```
  *.h5
  └── ${cohort_id}
      └── "ehr"
          ├── "hi"
          │	└── np.ndarray with a shape of (num_events, 3, max_length)
          ├── "time"
          │	└── np.ndarray with a shape of (num_events, )
          └── "label"
              └── binary label (0 or 1) for ${cohort_id} given the defined task
  ```
  * `${cohord_id}`: `${patient_id}_${cohort_number}`, standing for **N-th cohort in the patient**.
  * Numpy array under `"hi"`
    * `[:, 0, :]`: token input ids (i.e., `input_ids`) for the tokenized events.
    * `[:, 1, :]`: token type ids (i.e., `type_ids`) to distinguish where each input token comes from (special tokens such as `[CLS]` or `[SEP]`, column keys, or column values).
    * `[:, 2, :]`: tokens indicting digit places for number type tokens (i.e., `dpe_ids`). It assigns different ids to each of digit places for numeric (integer or float) items.
  * Numpy array under `"time"
    * Elapsed time in minutes from the first event to the last event. We do not this feature currently, but reserve it for future usage (e.g., developing a method to embed events with their temporal features).
* `*.tsv`
  ```
      patient_id  num_events
  0   10001472_0  13
  1   10002013_0  47
  2   10002013_1  46
  …   …           …
  ```

## STEP 3: Training a new model
We prepared example configuration files for various models and experimental setups.
For detailed configurations, please see [configs.py](src/genhpf/configs/configs.py) and each implemented source code (e.g., [genhpf.py](src/genhpf/models/genhpf.py)).

### Examples to process GenHPF dataset
### Train a new GenHPF model from scratch:
```bash
genhpf-train \
  dataset.data=??? \
  --config-dir ${GENHPF_DIR}/examples/train/genhpf \
  --config-name genhpf_hierarchical_scr
```
Note that you should fill in `dataset.data=???` with a path to the directory that contains the data manifest files (e.g., `train.tsv`, `valid.tsv`, etc.) for the processed GenHPF data.

### Pre-train and fine-tune a new GenHPF model:
For pre-training with SimCLR:
```bash
genhpf-train \
  dataset.data=??? \
  --config-dir ${GENHPF_DIR}/examples/pretrain/simclr/genhpf \
  --config-name genhpf_hierarchical_pt
```
For fine-tuning:
```bash
genhpf-train \
  dataset.data=??? \
  model.from_pretrained=${/path/to/the/pretrained/checkpoint.pt} \
  --config-dir ${GENHPF_DIR}/examples/train/genhpf \
  --config-name genhpf_hierarchical_ft
```

### Examples to process MEDS dataset
```bash
genhpf-train \
  dataset.data=??? \
  --config-dir ${GENHPF_DIR}/examples/train/genhpf \
  --config-name meds_hierarchical_scr
```
Note that you should fill in `dataset.data=???` with a path to the directory that contains the data manifest files (e.g., `train.tsv`, `tuning.tsv`, etc.) for the processed MEDS data (i.e., `$MEDS_OUTPUT_DIR`).

For doing inference on MEDS dataset while outputting prediction results to evaluate the model using [meds-evaluation](https://github.com/kamilest/meds-evaluation):
```bash
genhpf-test \
  dataset.data=??? \
  meds.output_predictions=true \
  meds.labels_dir=$MEDS_LABELS_DIR \
  meds.output_dir=$OUTPUT_DIR \
  checkpoint.load_checkpoint=${/path/to/the/trained/checkpoint.pt} \
  --config-dir ${GENHPF_DIR}/examples/test/genhpf \
  --config-name meds_hierarchical
```
This script will load the model weights from `${/path/to/the/trained/checkpoint.pt}`, process the data specified by `dataset.data`, and output the prediction results for the test subset as a single parquet file to `$OUTPUT_DIR` directory.
Note that the data directory `dataset.data` should contain the directory for the test data with its manifest file (e.g., `held_out/*.h5` with `held_out.tsv`), where the name of the test subset is specified by `dataset.test_subset` config.

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
