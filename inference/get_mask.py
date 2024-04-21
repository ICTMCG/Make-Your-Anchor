#!/usr/bin/python
# -*- encoding: utf-8 -*-

# from logger import setup_logger
from bisenet import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

import glob
import json

from tqdm import tqdm

import argparse

# [0, 'background', 1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye',
# 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',  10 'nose', 11 'mouth', 12 'u_lip',
# 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']

def dilate(img,  reverse=False):
    img = torch.from_numpy(img)
    mask = torch.ones_like(img)
    
    parsing = img
    mask = mask - ((parsing == 0).float())
    mask = mask - ((parsing == 14).float())
    mask = mask - ((parsing == 15).float())
    mask = mask - ((parsing == 16).float())
    mask = mask - ((parsing == 17).float())
    mask = mask - ((parsing == 18).float())
    
    kernel = np.ones((3,3), dtype=np.uint8) # origin maybe
    mask_numpy = mask.numpy()
    mask_numpy = cv2.dilate(mask_numpy, kernel, iterations=1)
    if reverse:
        mask_numpy = 1-mask_numpy
    mask_numpy = 255*mask_numpy

    return mask_numpy

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg', ifdilate=True, reverse=False):
    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)

    if ifdilate:
        vis_parsing_anno = dilate(vis_parsing_anno, reverse=reverse)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)


def evaluate(respth='./res/test_res', dspth='./data', cp='model_final_diss.pth', ifdilate=True, reverse=False):

    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    pathes = glob.glob( os.path.join(dspth,'*.jpg') ) + glob.glob( os.path.join(dspth,'*.png') )
    with torch.no_grad():
        for image_path in tqdm(pathes):
            image_path = os.path.basename(image_path)
            # print(image_path)
            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)


            vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path), ifdilate=ifdilate, reverse=reverse)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument( "--input_pth", type=str, default=None, required=True )
    parser.add_argument( "--mask_pth", type=str, default=None, required=True )

    args = parser.parse_args()

    ifdilate=True
    reverse=True

    respth=args.mask_pth
    dspth=args.input_pth

    evaluate(respth=respth, dspth=dspth, cp='79999_iter.pth', ifdilate=ifdilate, reverse=reverse)

