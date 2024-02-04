export MODEL=ziqingyang/chinese-llama-2-7b
export CLIP_MODEL=models/clip-vit-large-patch14-336-knowledge #openai/clip-vit-large-patch14-336
export MODEL_GROUP=$(echo $MODEL | cut -f1 -d/)
export MODEL_NAME=$(echo $MODEL | cut -f2 -d/)
export CLIP_MODEL_NAME=$(echo $CLIP_MODEL | cut -f2 -d/)
export OUTPUT_DIR=./checkpoints/llava-$MODEL_NAME-$CLIP_MODEL_NAME-knowledge
export EPOCH=5
#rm $OUTPUT_DIR/checkpoint-* -rf
#--model_name_or_path /root/mathlens_2.0/models/knowledge-$MODEL_NAME/model \
deepspeed scripts/train.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /root/mathlens_2.0/pretrained_local/llm/$MODEL_GROUP/$MODEL_NAME \
    --version v1 \
    --data_path data/knowledge/knowledge_finetune.train.json \
    --eval_data_path data/knowledge/knowledge_finetune.test.json \
    --image_folder data/knowledge/images \
    --vision_tower $CLIP_MODEL \
    --pretrain_mm_mlp_adapter ./models/llava-$MODEL_NAME-$CLIP_MODEL_NAME-pretrain-v1/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
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
    --eval_steps 20 \
    --evaluation_strategy steps \
    --per_device_eval_batch_size 8 \
    --report_to wandb