# GenHPF : General Healthcare Predictive Framework for Multi-task Multi-source Learning

GenHPF is a general healthcare predictive framework, which requires no medical domain knowledge and minimal preprocessing for multiple prediction tasks. 

Our framework presents a method for embedding any form of EHR systems for prediction tasks without requiring domain-knowledge-based pre-processing, such as medical code mapping and feature selection.  
				
This repository provides official Pytorch code to implement GenHPF, a general healthcare predictive framework.

# Getting started with GenHPF
## STEP 1 : Installation
Requirements

* [PyTorch](http://pytorch.org/) version >= 1.9.1
* Python version >= 3.8

## STEP 2: Prepare training data
First, download the required EHR datasets from these links: 
- [MIMIC-III](https://physionet.org/content/mimiciii/1.4/)
- [MIMIC-IV](https://physionet.org/content/mimiciv/2.0/)
- [eICU](https://physionet.org/content/eicu-crd/2.0/)

Note that you will need to request access for each dataset as they are publicly available electronic health records.

For preprocessing instructions, please refer to the README in the `preprocess` directory. 

The preprocessing will automatically handle the data preparation once you specify the raw data directory according to the preprocessing instructions.


## STEP 3. Training a new model
Other configurations will set to be default, which were used in the GenHPF paper.
$data should be set to 'mimic3' or 'eicu' or ‘mimic4’ 
`$model` should be set to one of [‘SAnD’, ‘Rajkomar’, ‘DescEmb’, ‘GenHPF’]

`$task` can be set to one of [mortality, long_term_mortality, los_3day, los_7day, readmission, final_acuity, imminent_discharge, diagnosis, creatinine, bilirubin, platelets, wbc] or multiple task.
The default setting is multi-task on all of tasks.

Note that `--input-path ` should be the root directory containing preprocessed data.
### Example
### Train a new GenHPF model:

```shell script
$ CUDA_VISIBLE_DEVICES=1 \
    python main.py \
    --input_path /path/to/data \
    --model_run GenHPF \
    --train_task scratch \
    --train_src $data \
    --pred_task $pred_task \
    --criterion prediction \
    --batch_size $batch_size \
    --world_size $world_size \
```
Note: if you want to train with baselines, set model_run as baseline model, one of (Rajikomar, DescEmb, SAnd).

### Pre-train GenHPF model:

```shell script
$ CUDA_VISIBLE_DEVICES=1 \
    python main.py \
    --input_path /path/to/data \
    --model_run GenHPF \
    --model GenHPF_simclr \
    --train_src $data \
    --train_task pretrain  \
    --pretrain_task $pretrain_task \
    --criterion $criterion \
    --batch_size $batch_size \
    --world_size $world_size \
    --valid_subset "" \
    
```

Note: if you want to train with pre-trained model, add command line parameters `--load_checkpoint` with directory of pre-trained model checkpoint and set `train_task` as `finetune`.

## Pooled learning 
```shell script
$ CUDA_VISIBLE_DEVICES=1 \
    python main.py \
    --input_path /path/to/data \
    --model_run GenHPF \
    --train_task scratch \
    --train_src mimiciii_eicu_mimiciv \
    --pred_task $pred_task \
    --criterion prediction \
    --batch_size $batch_size \
    --world_size $world_size \
```

Note: Please refer main.py argument ('train_src') for pooled learning.

## Transfer learning
```shell script
$ CUDA_VISIBLE_DEVICES=1 \
    python main.py \
    --input_path /path/to/data \
    --model GenHPF \
    –ratio 0 \
    --train_src mimiciii \
    --target_data eicu \
    --train_task finetune  \
    --pred_task $pred_task \
    --criterion prediction \
    --pretrain $scratch \
    --batch_size $batch_size \
    --world_size $world_size \
    --load_checkpoint $saved_ckpt \
```

Note that `--ratio` indicates proportion of target dataset for few-shot learning settings. (if ratio is set to zero, then it is zero shot learning) 

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
