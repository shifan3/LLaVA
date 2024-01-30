export MODEL=ziqingyang/chinese-llama-2-7b
export CLIP_MODEL=/root/LLaVA/checkpoints/clip-vit-large-patch14-336-finetuned/checkpoint-last/

export MODEL_GROUP=$(echo $MODEL | cut -f1 -d/)
export MODEL_NAME=$(echo $MODEL | cut -f2 -d/)

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /root/pretrained/$MODEL \
    --version v1 \
    --data_path data/LinkSoul/LLaVA-Instruct-150K/llava_instruct_80k.json.modified \
    --image_folder data/coco2017-train/train2017 \
    --vision_tower $CLIP_MODEL \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-$MODEL_NAME-clip-vit-large-patch14-336-pretrain/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-$MODEL_NAME-clip-vit-large-patch14-336-chat \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --image_aspect_ratio pad \
    --report_to none