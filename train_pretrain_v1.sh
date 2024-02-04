export MODEL=ziqingyang/chinese-llama-2-7b

export CLIP_MODEL=models/clip-vit-large-patch14-336-knowledge
export MODEL_GROUP=$(echo $MODEL | cut -f1 -d/)
export MODEL_NAME=$(echo $MODEL | cut -f2 -d/)
export CLIP_MODEL_NAME=$(echo $CLIP_MODEL | cut -f2 -d/)
export OUTPUT_DIR=./checkpoints/llava-$MODEL_NAME-$CLIP_MODEL_NAME-pretrain-v1
#!/bin/bash

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /root/mathlens_2.0/pretrained_local/llm/$MODEL \
    --version plain \
    --data_path data/knowledge/knowledge_pretrain.json \
    --image_folder data/knowledge/images/ \
    --vision_tower $CLIP_MODEL \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb


