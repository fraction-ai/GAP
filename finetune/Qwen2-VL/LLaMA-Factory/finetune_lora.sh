#!/bin/bash

GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
    "

torchrun $DISTRIBUTED_ARGS src/train.py \
    --deepspeed ds_config_zero2.json \
    --stage sft \
    --do_train \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --dataset gap_vqa \
    --template qwen2_vl \
    --finetuning_type lora \
    --lora_target all \
    --output_dir output/output_lora \
    --overwrite_cache \
    --overwrite_output_dir \
    --warmup_ratio 0.01 \
    --weight_decay 0.1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --ddp_timeout 9000000 \
    --learning_rate 4e-5 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --cutoff_len 2048 \
    --save_steps 100 \
    --plot_loss \
    --num_train_epochs 5 \
    --bf16 true \
    --val_size 0.1 \
    --eval_strategy steps \
    --eval_steps 100 \
    --report_to wandb \
    --run_name qwen2vl_7B_gap_vqa_4e-5
