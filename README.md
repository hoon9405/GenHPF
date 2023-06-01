# GenHPF : General Healthcare Predictive Framework with Zero Domain Knowledge

GenHPF is a general healthcare predictive framework, which requires no medical domain knowledge and minimal preprocessing for multiple prediction tasks. Given any time-series EHR, 		 	 	 			
					
Our framework presents a method for embedding any form of EHR systems for prediction tasks with- out requiring domain-knowledge-based pre-processing, such as medical code mapping and feature selection.  
				
This repository provides official Pytorch code to implement GenHPF, a general healthcare predictive framework.

# Getting started with GenHPF
## STEP 1 : Installation
Requirements

* [PyTorch](http://pytorch.org/) version >= 1.9.1
* Python version >= 3.8

## STEP 2: Prepare training data
First, download the dataset from these links: 
	[MIMIC-III](https://physionet.org/content/mimiciii/1.4/)
[MIMIC-IV](https://physionet.org/content/mimiciv/2.0/)
[eICU](https://physionet.org/content/eicu-crd/2.0/)
[ccs_multi_dx_tool_2015](https://www.hcup-us.ahrq.gov/toolssoftware/ccs/Multi_Level_CCS_2015.zip)
[icd10cmtoicd9gem](https://data.nber.org/gem/icd10cmtoicd9gem.csv)
	
!

Note that you will need to get access for each dataset, publicly available electronic health records. 

Second, for preparation your data, make directory structure like below:
```
dataset_path
├─ mimic3
│  ├─ ADMISSIONS.csv
│  ├─ PATIENTS.csv
│  ├─ ICUSYAYS.csv
│  ├─ LABEVENTES.csv
│  ├─ PRESCRIPTIONS.csv
│  ├─ PROCEDURES.csv
│  ├─ INPUTEVENTS_CV.csv
│  ├─ INPUTEVENTS_MV.csv
│  ├─ D_ITEMDS.csv
│  ├─ D_ICD_PROCEDURES.csv
│  └─ D_LABITEMS.csv
├─ eicu
│  ├─ diagnosis.csv
│  ├─ infusionDrug.csv
│  ├─ lab.csv
│  ├─ medication.csv
│  └─ patient.csv
├─ mimci4
│  ├─ admissions.csv
│  ├─ …
│  └─ d_labitems.csv
├─ ccs_multi_dx_tool_2015.csv
└─ icd10cmtoicd9gem.csv

```
Then run preprocessing code
```shell script
$ bash preprocess/preprocess_run.sh 
    $destination_directory
    $data_directory 
```
Note that pre-processing takes about 6hours in 128 cores of AMD EPYC 7502 32-Core Processor, and requires 180GB of RAM.


STEP 3. Training a new model
Other configurations will set to be default, which were used in the GenHPF paper.
$data should be set to 'mimic3' or 'eicu' or ‘mimic4’ 
`$model` should be set to one of [‘SAnD’, ‘Rajkomar’, ‘DescEmb’, ‘GenHPF’]

`$task` can be set to one of [mortality, long_term_mortality, los_3day, los_7day, readmission, final_acuity, imminent_discharge, diagnosis, creatinine, bilirubin, platelets, wbc] or multiple task.
The default setting is multi-task on all of tasks.

Note that `--input-path ` should be the root directory containing preprocessed data.
### Example
### Train a new GenHPF model:

```shell script
$ python main.py \
    --input_path /path/to/data \
    --model_run GenHPF \
    --train_task scratch \
    --src_data $data \
    --pred_task $pred_task \
    --batch_size $batch_size \
    --device_num $device_num \
```
Note: if you want to train with baselines, set model_run as baseline model, one of (Rajikomar, DescEmb, SAnd).

### Pre-train GenHPF model:

```shell script
$ python main.py \
    --input_path /path/to/data \
    --model_run GenHPF \
    --model GenHPF_simclr \
    --src_data $data \
    --train_task pretrain  \
    --pretrain_task $pretrain_task \
    --batch_size $batch_size \
    --device_num $device_num \
```

Note: if you want to train with pre-trained model, add command line parameters `--load_checkpoint` with directory of pre-trained model checkpoint and set `train_task` as `finetune`.

## Pooled learning 
```shell script
$ python main.py \
    --input_path /path/to/data \
    --model_run GenHPF \
    --train_task scratch \
    --src_data mimiciii_eicu_mimiciv \
    --pred_task $pred_task \
    --batch_size $batch_size \
    --device_num $device_num \
```

Note: Please refer main.py argument ('src_data') for pooled learning.

## Transfer learning
```shell script
$ python main.py \
    --input_path /path/to/data \
    --model GenHPF \
    –ratio 0 \
    --src_data mimiciii \
    --target_data eicu \
    --train_task finetune  \
    --pred_task $pred_task \
    --pretrain $scratch \
    --batch_size $batch_size \
    --device_num $device_num \
    --load_checkpoint $saved_ckpt \
```

Note that `--ratio` indicates proportion of target dataset for few-shot learning settings. (if ratio is set to zero, then it is zero shot learning) 


# License
This repository is MIT-lincensed.
