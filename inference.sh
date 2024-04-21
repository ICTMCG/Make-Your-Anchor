cd inference

export CUDA_VISIBLE_DEVICES=4

## Please fill the parameters here
# path to the body model folder
body_weight_dir=./checkpoints/seth/body
# path to the head model folder
head_weight_dir=./checkpoints/seth/head
# path to the input poses
body_input_dir=./samples/poses/seth1
# path to the reference body appearance
body_prompt_img_pth=./samples/appearance/body.png
# path to the reference head appearance
head_prompt_img_pth=./samples/appearance/head.png

# pipe_path=runwayml/stable-diffusion-v1-5
# pipe_inpainting_path=runwayml/stable-diffusion-inpainting

body_unet_path=${body_weight_dir}/unet
body_controlent_path=${body_weight_dir}/controlnet
body_cfg=7.5
body_condition_scale=2.

head_unet_pth=${head_weight_dir}/unet
head_controlnet_path=${head_weight_dir}/controlnet
head_cfg=3.5
head_condition_scale=1.

save_root=./samples/output


fps=30

body_save_dir=${save_root}/body_output
body_headcrop_root=${body_save_dir}_meshcrop
head_dir=${body_headcrop_root}/raw_aligned
head_mask_dir=${body_headcrop_root}/raw_aligned_mask
head_input_dir=${body_headcrop_root}/aligned
head_matrix_dir=${body_headcrop_root}/matrix
head_save_dir=${body_headcrop_root}/face_output

final_save_dir=${save_root}/final_output

# inference on body
python inference_body.py \
    --unet_path $body_unet_path \
    --controlnet_path $body_controlent_path \
    --input_dir $body_input_dir \
    --save_dir $body_save_dir \
    --prompt_img_pth $body_prompt_img_pth \
    --CFG $body_cfg \
    --condition_scale $body_condition_scale \
    # --pipe_path $pipe_path

ffmpeg -r 30 -i $body_save_dir/%06d.png -q:v 0 -pix_fmt yuv420p $body_save_dir.mp4

# face alignment
python face_alignment.py \
    --imgdir_pth $body_input_dir \
    --raw_imgdir_pth $body_save_dir \
    --results_dir $body_headcrop_root \
    --crop_size 512

# get face mask
python get_mask.py \
    --input_pth $head_dir \
    --mask_pth $head_mask_dir


# inference on face inpainting
python inference_face_inpainting.py \
    --unet_path $head_unet_pth \
    --controlnet_path $head_controlnet_path \
    --input_dir $head_input_dir \
    --face_dir $head_dir \
    --mask_dir $head_mask_dir \
    --save_dir $head_save_dir \
    --prompt_img_pth $head_prompt_img_pth \
    --CFG $head_cfg \
    --condition_scale $head_condition_scale \
    --batch_size 30 \
    # --pipe_pth $pipe_inpainting_path


# face blending
python face_blending.py \
    --body_dir $body_save_dir \
    --face_dir $head_save_dir \
    --matrix_dir $head_matrix_dir \
    --save_dir $final_save_dir \
    --crop_size 512 


ffmpeg -r 30 -i $final_save_dir/%06d.png -q:v 0 -pix_fmt yuv420p $final_save_dir.mp4