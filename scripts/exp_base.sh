export CUDA_VISIBLE_DEVICES=1

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="./data/FluxCount/subsets/apple_orange_peach/"

accelerate launch ./src/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-06 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="exp_base/" \
  --instance_encoder_config "{}" \
  --attn_loss_config "{'cross_attn_loss_scale': 0.01, 'clean_cross_attn_loss_scale': 0.001, 'self_attn_loss_scale': 0.01}" \
  --report_to "wandb"