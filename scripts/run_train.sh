for ws in 3
do
    echo window size: $ws
    python main.py \
            --mode train \
            --random_seed 227182 \
            --window_size $ws \
            --batch_size 3000 \
            --save_checkpoint_steps 10000 \
            --train_steps 180000 \
            --valid_steps 20000 \
            --visible_gpus 1 \
            --backbone_type bert \
            --add_transformer False \
            --finetune_bert True \
            --classifier_type conv \
            --lr 0.002 \
            --warmup_steps 10000 \
            --model_index NF0$ws
done
# for index in `seq 0 2 10`
# do
#     echo print_$index.pt
# done