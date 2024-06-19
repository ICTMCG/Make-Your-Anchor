cd train

export CUDA_VISIBLE_DEVICES=0
export MODEL_DIR="runwayml/stable-diffusion-inpainting"

# load weights from stage-1
unet_model_name_or_path="./checkpoints/pre-trained_weight/head/unet"
controlnet_model_name_or_path="./checkpoints/pre-trained_weight/head/unet"

export OUTPUT_DIR="path/to/save_folder"
json_file=./train_data/head_train.json

# for validation
val_img1=path/to/head_pose_1
val_img2=path/to/head_pose_2
val_img3=path/to/head_pose_3
val_img4=path/to/head_pose_4

val_mask1=path/to/head_mask_1
val_mask2=path/to/head_mask_2
val_mask3=path/to/head_mask_3
val_mask4=path/to/head_mask_4

reference_img=path/to/head_ref

accelerate launch --main_process_port 65535 train_head.py \
 --resume_from_checkpoint latest \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --controlnet_model_name_or_path=$controlnet_model_name_or_path \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=json \
 --dataset_config_name $json_file \
 --image_column target \
 --conditioning_image_column source \
 --caption_column prompt \
 --resolution=256 \
 --learning_rate=1e-5 \
 --validation_image $val_img1 $val_img2 $val_img3 $val_img4 \
 --validation_mask $val_mask1 $val_mask2 $val_mask3 $val_mask4 \
 --reference_image $reference_img $reference_img $reference_img $reference_img \
 --train_batch_size=4 \
 --enable_xformers_memory_efficient_attention \
 --tracker_project_name train_head \
 --checkpointing_steps 10000 \
 --validation_steps 1000 \
 --num_train_epochs 60