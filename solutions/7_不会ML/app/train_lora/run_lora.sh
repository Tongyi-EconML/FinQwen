#!/bin/bash

# 清除屏幕
clear

N_GPU_PER_NODE=2
N_NODE=1

MODEL='qwen/Qwen-7B-Chat'
MODEL_SAVE_DIR="./model_save/ner_adalora"
DATA='/root/autodl-tmp/NER_lora_train.json'
NUM_TRAIN_EPOCHS=2
BATCH_SIZE=8
SAVE_STEPS=50
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE=3e-4
# VAL_DATA="/root/autodl-tmp/bojin_competition/peft_model/sft_data/NER_lora_test_3.json"
DS_CONFIG_PATH="lora_config.json"
LORA_RANK=20
LORA_ALPHA=8
LORA_DROPOUT=0.01
LORA_BIAS='none'
SYSTEM_MESSAGE='你是一个NER智能体'
SAVE_TOTAL_LIMIT=1

accelerate launch \
  --num_machines $N_NODE \
  --num_processes $(($N_NODE*$N_GPU_PER_NODE)) \
  --use_deepspeed \
  --deepspeed_multinode_launcher 'standard' \
  --zero_stage 2 \
  --offload_optimizer_device 'cpu' \
  --offload_param_device 'none' \
  --gradient_accumulation_steps 1 \
  --gradient_clipping 1.0 \
  --zero3_init_flag false \
  --zero3_save_16bit_model false \
  --main_training_function 'main' \
  --mixed_precision 'bf16' \
  --dynamo_backend 'no' \
  --same_network \
  finetune.py \
  --model_name_or_path $MODEL \
  --data_path $DATA \
  --bf16 True \
  --output_dir $MODEL_SAVE_DIR \
  --num_train_epochs $NUM_TRAIN_EPOCHS \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps $SAVE_STEPS \
  --save_total_limit $SAVE_TOTAL_LIMIT \
  --learning_rate $LEARNING_RATE \
  --weight_decay 0.1 \
  --adam_beta2 0.95 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --report_to "none" \
  --model_max_length 512 \
  --lazy_preprocess True \
  --use_lora  \
  --system_message $SYSTEM_MESSAGE \
  --lora_r $LORA_RANK\
  --lora_alpha $LORA_ALPHA\
  --lora_dropout $LORA_DROPOUT\
  --lora_bias $LORA_BIAS\
  --gradient_checkpointing \
  --deepspeed ${DS_CONFIG_PATH} 
