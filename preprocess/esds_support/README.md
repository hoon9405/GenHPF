# Pre-processing for ESDS-Compliant Dataset
We officially support ESDS-compliant Dataset to be processed for the GenHPF model using a set of transformation functions that converts ESDS-compliant dataset into GenHPF-formatted data that can be directly forwarded to the current GenHPF trainer and model.
This currently includes pre-processing for both the hierarchical structure where the dataset has a 3-dimensinal shape of (Number of samples, Number of Events, Event Tokens), and the flattened structure where the dataset has a 2-dimensional shape of (Number of samples, Flattened Event Tokens (Number of Events * Event Tokens, potentially)).

# Getting Started
run:
```shell script
$ python esds2genhpf.py \
    --dataset_path /path/to/your/ESDS/dataset \
    --output_path /path/to/output \
    --output_name $output_name
```
Note that this script currently assumes you are loading an [Event Stream GPT](https://github.com/mmcdermott/EventStreamGPT) so you may need to modify the script to fit your dataset format. The example guideline for running [Event Stream GPT](https://github.com/mmcdermott/EventStreamGPT) pipeline to get ESDS-compliant dataset can be also found below.  

You also should be aware of that this script does not process labels for tasks required for the GenHPF trainer.
To process the data with labels, you should add new columns in `$output_name.csv` where the column name is the task name and the values are labels corresponded with `icustay_id` for each row.  
For example, if you want to run experiments for mortality prediction task, then your `$output_name.csv` should look like:
```
icustay_id, split_1,    mortality
000001      train       0
000002      train       1
000003      valid       1
000004      test        0
...         ...         ...
```

# Example with [Event Stream GPT](https://github.com/mmcdermott/EventStreamGPT)
For those who want to get ESDS-compliant dataset, we are providing an example guideline to run [Event Stream GPT](https://github.com/mmcdermott/EventStreamGPT) pipeline for three different public EHR datasets: [MIMIC-III](https://mimic.mit.edu/docs/iii/), [MIMIC-IV](https://mimic.mit.edu/docs/iv/), and [eICU](https://eicu-crd.mit.edu/about/eicu/).

## Run Event Stream GPT Pipeline

### Preliminary Steps

1. Set up MIMIC-III, MIMIC-IV, and eICU on your local machine as a postgresql database (see guidelines for [MIMIC-III](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iii/buildmimic/postgres), [MIMIC-IV](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/buildmimic/postgres), and [eICU](https://github.com/MIT-LCP/eicu-code/tree/main/build-db/postgres)).
2. Clone the [Event Stream GPT](https://github.com/mmcdermott/EventStreamGPT/tree/v0.5) library on the v0.5 branch, install its dependencies (run `pip install -e .`) in the root directory of the repo.
3. Copy `example/.env.example` to a file named `example/.env` and set the environment variables appropriately.
    * `PROJECT_DIR`: This is a directory to where you want to output the processed datasets.
    * `PROJECT_DATA_DIR`: This should be set to the same with `PROJECT_DIR`.
    * `EVENT_STREAM_PATH`: This is the path at which you cloned the Event Stream GPT repository.
4. install [rootutils](https://github.com/ashleve/rootutils) (`pip install rootutils`).

### Run for MIMIC-III
```shell script
$ cd example
$ ./build_esds_dataset.sh dataset_genhpf_mimiciii $esgpt_dir do_overwrite=True
```

### Run for MIMIC-IV
```shell script
$ cd example
$ ./build_esds_dataset.sh dataset_genhpf_mimiciv $esgpt_dir do_overwrite=True
```

### Run for eICU
```shell script
$ cd example
$ ./build_esds_dataset.sh dataset_genhpf_eicu $esgpt_dir do_overwrite=True
```
`$esgpt_dir` is a directory to where your output will be located.

### Convert to GenHPF-formatted Dataset
After building the ESDS-compliant datasets, run:
```shell script
$ python esds2genhpf.py \
    --dataset_path $esgpt_dir/ESDS \
    --output_path $output_dir \
    --output_name $output_name
```
`$output_dir` is a directory to where the converted GenHPF-formatted dataset will be saved.  
`$output_name` is a file name of the converted GenHPF-formatted dataset without extension.