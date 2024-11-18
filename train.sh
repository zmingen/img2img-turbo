# 定义共同参数
PRETRAINED_MODEL_PATH="output/pix2pix_turbo/angel_0_go/checkpoints/model_2501.pkl"
RESOLUTION=512
BATCH_SIZE=2
VIZ_FREQ=25
NUM_EPOCHS=4
LAMBDA_CLIPSIM=-1
REPORT_TO="wandb"

# 循环处理不同的输出目录和数据集
for i in {2..5}; do
  OUTPUT_DIR="output/pix2pix_turbo/angel_$i"
  DATASET_FOLDER="data/oct/angel_$i"
  PROJECT_NAME="pix2pix_turbo_angel_$i"

  # 执行 accelerate 命令
  accelerate launch src/train_pix2pix_turbo.py \
    --pretrained_model_name_or_path="$PRETRAINED_MODEL_PATH" \
    --output_dir="$OUTPUT_DIR" \
    --dataset_folder="$DATASET_FOLDER" \
    --resolution=$RESOLUTION \
    --train_batch_size=$BATCH_SIZE \
    --enable_xformer \
    --viz_freq=$VIZ_FREQ \
    --track_val_fid \
    --report_to="$REPORT_TO" \
    --tracker_project_name="$PROJECT_NAME" \
    --num_training_epochs=$NUM_EPOCHS \
    --lambda_clipsim=$LAMBDA_CLIPSIM
    --checkpointing_steps=200
done
