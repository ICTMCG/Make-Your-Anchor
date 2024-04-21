import os
import cv2
from glob import glob
from diffusers import ControlNetModel, UNet2DConditionModel, DDIMScheduler

from pipelines.pipeline_stable_diffusion_controlnet_inpaint import StableDiffusionControlNetInpaintPipeline

from PIL import Image
import torch
from tqdm import tqdm

from pytorch_lightning import seed_everything

from transformers import CLIPProcessor, CLIPVisionModel

import argparse



parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
parser.add_argument(
    "--pipe_pth",
    type=str,
    default='runwayml/stable-diffusion-inpainting',
)
parser.add_argument( "--unet_path", type=str, default=None, required=True )
parser.add_argument( "--controlnet_path", type=str, default=None, required=True )
parser.add_argument( "--input_dir", type=str, default=None, required=True )
parser.add_argument( "--face_dir", type=str, default=None, required=True )
parser.add_argument( "--mask_dir", type=str, default=None, required=True )
parser.add_argument( "--save_dir", type=str, default=None, required=True )
parser.add_argument( "--prompt_img_pth", type=str, default=None, required=True )
parser.add_argument( "--CFG", type=float, default=1.5 )
parser.add_argument( "--condition_scale", type=float, default=1. )
parser.add_argument( "--batch_size", type=int, default=30 )

args = parser.parse_args()

# Load CLIP Image Encoder
clip_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
clip_encoder.requires_grad_(False)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

pipe_pth = args.pipe_pth

controlnet = ControlNetModel.from_pretrained( args.controlnet_path )
unet = UNet2DConditionModel.from_pretrained( args.unet_path )


pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    pipe_pth, controlnet=controlnet, safety_checker=None, device='cuda'
)
pipe.unet = unet


# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
ddim_inv_scheduler = DDIMScheduler.from_pretrained(pipe_pth, subfolder='scheduler')
pipe.scheduler = ddim_inv_scheduler

# Remove if you do not have xformers installed
# see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
# for installation instructions
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

def encode_img_clip(image):
    inputs = clip_processor(images=image, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}
    clip_image_embeddings = clip_encoder(**inputs).last_hidden_state.cuda()
    return clip_image_embeddings


def image_process(img_pth):
    try:
        img = cv2.imread(img_pth)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
    except Exception as e:
        print(img, e)
        raise e
    return Image.fromarray(img)

def inference_batch(input_pths, save_pths, mask_pths, face_pths, latents=None, prompt_embeds=None, num_samples=1):
    control = [image_process(i) for i in input_pths]
    mask = [image_process(i) for i in mask_pths]
    faces = [image_process(i) for i in face_pths]
    
    
    generator = torch.manual_seed(0) # for GPU

    images = pipe(
                  image=faces, 
                  control_image=control,
                  mask_image=mask,
                  num_inference_steps=20, 
                  guidance_scale=args.CFG,
                  controlnet_conditioning_scale=args.condition_scale,
                  generator=generator, # to reproduce pipeline
                  latents=latents,
                  prompt_embeds=prompt_embeds,
                ).images

    for image, s_pth in zip(images, save_pths):
        image.save(s_pth)
    
if __name__ == '__main__':

    input_dir = args.input_dir
    face_dir = args.face_dir
    mask_dir = args.mask_dir
    save_dir = args.save_dir

    prompt = 'male host in formal wear live'
    
    os.makedirs(save_dir, exist_ok=True)

    prompt_img_pth = args.prompt_img_pth

    prompt_img = image_process(prompt_img_pth)
    prompt_embeds = encode_img_clip(prompt_img)
    
    img_pths = glob(os.path.join(input_dir,'*.jpg'))
    img_pths.sort()
    print(img_pths)

    seed_everything(999)

    batch_size = args.batch_size
    batch={}
    batch['img_pth']=[]
    batch['save_pth']=[]
    batch['mask_pth']=[]
    batch['face_pth']=[]
    batch_num=0


    latents = torch.randn((1, 4, 32, 32), device="cuda")
    latents_inp = latents.repeat(len(range(batch_size)), 1, 1, 1)
    prompt_embeds = prompt_embeds.cuda()
    
    for img_pth in tqdm(img_pths):
        img_name = os.path.basename(img_pth)
        save_pth = os.path.join(save_dir, img_name[:-4]+'.png')
        mask_pth = os.path.join(mask_dir, img_name[:-4]+'.png')
        face_pth = os.path.join(face_dir, img_name[:-4]+'.png')

        batch['img_pth'].append(img_pth)
        batch['save_pth'].append(save_pth)
        batch['mask_pth'].append(mask_pth)
        batch['face_pth'].append(face_pth)
        # print(face_pth)
        batch_num += 1
        if batch_num >= batch_size:
            prompt_embeds_input = prompt_embeds.repeat(batch_num, 1, 1)
            inference_batch(batch['img_pth'], batch['save_pth'], batch['mask_pth'], batch['face_pth'], latents=latents_inp, prompt_embeds=prompt_embeds_input)
            batch = {}
            batch['img_pth']=[]
            batch['save_pth']=[]
            batch['mask_pth']=[]
            batch['face_pth']=[]
            batch_num = 0
            

    if batch_num != 0:
        prompt_embeds_input = prompt_embeds.repeat(batch_num, 1, 1)
        latents_inp = latents.repeat(len(range(batch_num)), 1, 1, 1)
        inference_batch(batch['img_pth'], batch['save_pth'], batch['mask_pth'], batch['face_pth'], latents=latents_inp, prompt_embeds=prompt_embeds_input)
