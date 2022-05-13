for ws in 3
do
    echo window size: $ws
    python main.py \
            --mode train \
            --random_seed 227182 \
            --window_size $ws \
            --batch_size 3000 \
            --save_checkpoint_steps 10000 \
            --train_steps 50000 \
            --valid_steps 10000 \
            --visible_gpus 0 \
            --backbone_type bert \
            --add_transformer False \
            --finetune_bert True \
            --classifier_type conv \
            --lr 0.002 \
            --warmup_steps 10000 \
            --model_index A0$ws
done