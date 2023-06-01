device_num=$1
world_size=$2
train_src=$3
batch_size=$4
seed=$5

OMP_NUM_THREADS=16 \
CUDA_VISIBLE_DEVICES=$device_num \
python3 ../../main.py \
--train_task pretrain \
--train_src $train_src \
--pretrain_task simclr \
--model_run GenHPF \
--model GenHPF_simclr \
--batch_size $batch_size \
--world_size $world_size \
--criterion simclr \
--best_checkpoint_metric loss \
--max_epoch 200 \
--seed "1,2,3,4,5" \
--valid_subset "" \
--wandb \
