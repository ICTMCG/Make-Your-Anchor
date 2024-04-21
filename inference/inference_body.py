import os
import argparse
import torch

from PIL import Image

from diffusers import DDIMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from models.pipeline_controlvideo import ControlVideoPipeline
from models.util import save_image
from models.unet import UNet3DConditionModel
from models.controlnet import ControlNetModel3D

from glob import glob

from transformers import CLIPTokenizer, CLIPProcessor, CLIPVisionModel


import argparse

# Load CLIP Image Encoder
clip_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
clip_encoder.requires_grad_(False)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def encode_img_clip(image):
    inputs = clip_processor(images=image, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}
    clip_image_embeddings = clip_encoder(**inputs).last_hidden_state.cuda()
    return clip_image_embeddings


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=512, help="Height of synthesized video, and should be a multiple of 32")
    parser.add_argument("--width", type=int, default=512, help="Width of synthesized video, and should be a multiple of 32")
    parser.add_argument("--smoother_steps", nargs='+', default=[19, 20], type=int, help="Timesteps at which using interleaved-frame smoother")
    parser.add_argument("--is_long_video", action='store_true', help="Whether to use hierarchical sampler to produce long video")
    parser.add_argument("--seed", type=int, default=42, help="Random seed of generator")
    

    parser.add_argument(
        "--pipe_path",
        type=str,
        default='runwayml/stable-diffusion-v1-5',
    )
    parser.add_argument( "--unet_path", type=str, default=None, required=True )
    parser.add_argument( "--controlnet_path", type=str, default=None, required=True )
    parser.add_argument( "--input_dir", type=str, default=None, required=True )
    parser.add_argument( "--save_dir", type=str, default=None, required=True )
    parser.add_argument( "--prompt_img_pth", type=str, default=None, required=True )
    parser.add_argument( "--CFG", type=float, default=1.5 )
    parser.add_argument( "--condition_scale", type=float, default=1. )

    parser.add_argument("--num_steps", type=int, default=20)

    parser.add_argument("--ws", type=int, default=16)
    parser.add_argument("--os", type=int, default=4)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    device = "cuda"
    sd_path = args.pipe_path
    
    # Height and width should be a multiple of 32
    args.height = (args.height // 32) * 32    
    args.width = (args.width // 32) * 32    

    controlnet_pth = args.controlnet_path
    unet_pth = args.unet_path

    tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder").to(dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae").to(dtype=torch.float16)
    unet = UNet3DConditionModel.from_pretrained_2d(unet_pth).to(dtype=torch.float16)
    controlnet = ControlNetModel3D.from_pretrained_2d(controlnet_pth).to(dtype=torch.float16)


    scheduler=DDIMScheduler.from_pretrained(sd_path, subfolder="scheduler")

    pipe = ControlVideoPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            controlnet=controlnet, scheduler=scheduler,
        )
    pipe.enable_vae_slicing()
    pipe.enable_xformers_memory_efficient_attention()
    pipe.to(device)

    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)

    input_dir = args.input_dir
    prompt_img_pth = args.prompt_img_pth
    save_dir= args.save_dir
    
    prompt_img = Image.open(prompt_img_pth)
    prompt_embeds = encode_img_clip(prompt_img)


    img_pths = glob(input_dir+'/*.png')
    img_pths.sort()

    pil_annotation = [ Image.open(img) for img in img_pths]

    video_length = len(pil_annotation)

    # Step 3. inference
    sample = pipe.generate_long_video_slidingwindow_overlap(
                video_length=video_length, frames=pil_annotation, 
                num_inference_steps=args.num_steps, smooth_steps=args.smoother_steps, window_size=args.ws, window_overlap=args.os,
                generator=generator, 
                guidance_scale=args.CFG,
                prompt_embeds=prompt_embeds,
                controlnet_conditioning_scale=args.condition_scale,
                width=args.width, 
                height=args.height,
            ).videos
    os.makedirs(save_dir, exist_ok=True)

    for idx, img_pth in enumerate(img_pths):
        img_name = os.path.basename(img_pth)
        save_pth = os.path.join(save_dir, img_name)
        output_image = sample[:,:,]
        save_image(sample[:,:,idx,:,:].unsqueeze(2), save_pth)