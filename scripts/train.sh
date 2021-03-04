#!/bin/zsh

python train.py \
    -d '()' \
    --batch_size 512 \
    --raw_data data_tot_mat.txt \
    --data_root ./raw_data \
    --dataset data_tot \
    --workers 2 \
    --num_input 5 \
    --num_output 2 \
    --num_h 2 \
    --lr 0.01 \
    --neurons_per_hlayer 5 15  \
    --epochs_per_decay 20 40 45 60 80 90 \
    --total_epochs 5 \
    --epochs_per_val 1 \
    --epochs_per_save 1 \
    --portion_train 0.7 \
    --portion_val 0.15 \
    --run 50 \
    --load_model_weight False \
    --ckpt_file "" \
