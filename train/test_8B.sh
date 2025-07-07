#!/bin/bash

# MONAI / VILA M3 Fine-tuning Script (SLAKE only, Single RTX A6000 GPU)
export DISABLE_CLEARML=true
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# 환경 설정
MASTER_ADDR=127.0.0.1
MASTER_PORT=25001
GPUS_PER_NODE=2 # 사용할 GPU 수
NODES=1
CURRENT_RANK=0
n_node=1
bs=2    # Batch 사이즈

# 사용자 경로 설정
VILA_CODE_PATH="/home/hufsaim/VLM/VLM/thirdparty/VILA"
STAGE2_PATH="MONAI/Llama3-VILA-M3-8B"
OUTPUT_DIR="/home/hufsaim/VLM/VLM/m3/train/checkpoints/lorattcc"
OUTPUT_DIR2="/home/hufsaim/VLM/VLM/m3/train/checkpoints/lorattcc2"
WANDB_API_KEY="548576cef23d93fa2dc796520d3ba5f38c909d2d"

# Conda 환경 및 코드 경로 설정
source /root/miniconda3/bin/activate
source /root/.bashrc
cd $VILA_CODE_PATH
conda activate vila
export PYTHONPATH=$VILA_CODE_PATH
echo "Using python: $(which python)"
echo "PYTHONPATH is $PYTHONPATH"
wandb login $WANDB_API_KEY

# export CUDA_VISIBLE_DEVICES=0, 1

HEALTHCARE_DS=$(for i in {1..3}; do echo -n tumor_classification+; done)
HEALTHCARE_DS=${HEALTHCARE_DS%+}

# 학습 실행
torchrun --nnodes=$n_node --nproc_per_node=$GPUS_PER_NODE --master_port=$MASTER_PORT \
    --master_addr $MASTER_ADDR --node_rank=$CURRENT_RANK \
    llava/train/train_mem.py \
    --model_name_or_path $STAGE2_PATH \
    --version llama_3 \
    --data_mixture ${HEALTHCARE_DS} \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --lora_enable True \
    --lora_llm True \
    --lora_vt False \
    --tune_vision_tower False \
    --tune_mm_projector True \
    --tune_language_model False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --ddp_find_unused_parameters True \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --vflan_no_system_prompt True \
    --report_to wandb