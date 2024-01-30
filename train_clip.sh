export CLIP_MODEL=openai/clip-vit-large-patch14 #openai/clip-vit-large-patch14-336
export CLIP_MODEL_NAME=$(echo $CLIP_MODEL | cut -f2 -d/)
python3.10 scripts/clip/train_clip_prepare.py $CLIP_MODEL
OUTPUT_DIR=./checkpoints/$CLIP_MODEL_NAME-knowledge
python3.10 scripts/clip/train_clip1.py \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $CLIP_MODEL \
    --image_column image_path \
    --caption_column caption \
    --remove_unused_columns=False \
    --do_train  --do_eval \
    --per_device_train_batch_size="32" \
    --per_device_eval_batch_size="8" \
    --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
    --overwrite_output_dir \
    --report_to wandb \
    --logging_steps 20 \
    --dataloader_drop_last True \
    --eval_steps 500 \
    --evaluation_strategy steps \
    --bf16 \
    --save_total_limit 3 \
    #--resume_from_checkpoint $OUTPUT_DIR/checkpoint-500
    #--deepspeed ./scripts/zero3_clip.json \