import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from util.reverse2original import reverse2wholeimage
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet

import argparse


def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

def imgdir_inverse_align_bylist(
    body_dir,
    face_dir,
    matrix_dir,
    save_dir,      # for save dir
    crop_size=224, 
    use_mask =False
):
    spNorm =SpecificNorm()
    if use_mask:
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
        net.load_state_dict(torch.load(save_pth))
        net.eval()
    else:
        net = None
    
    os.makedirs(save_dir, exist_ok=True)

    length = len(os.listdir(face_dir))

    for frame_idx in tqdm(range(1,length+1)):
        body_pth = os.path.join(body_dir, '{:06d}.png'.format(frame_idx))
        face_pth = os.path.join(face_dir, '{:06d}.png'.format(frame_idx))
        matrix_pth = os.path.join(matrix_dir, '{:06d}.npy'.format(frame_idx))


        body = cv2.imread(body_pth)
        img = body

        m = np.load(matrix_pth)
        img_mat_list = [m]

        aligned_pth=face_pth
        img_align_crop = cv2.imread(aligned_pth)
        img_align_crop = cv2.resize(img_align_crop, (512,512))
        # BGR TO RGB
        img_align_crop_tenor = _totensor(cv2.cvtColor(img_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()            
        img_align_crop_tenor_list = [img_align_crop_tenor]
        
        edit_align = cv2.imread(face_pth)
        # BGR TO RGB
        edit_align = cv2.resize(edit_align, (crop_size,crop_size))

        edit_align_tenor = _totensor(cv2.cvtColor(edit_align,cv2.COLOR_BGR2RGB))[None,...].cuda()
        edit_align_tenor = edit_align_tenor.squeeze(0)
        edit_result_list = [edit_align_tenor]

        reverse2wholeimage(img_align_crop_tenor_list, edit_result_list, img_mat_list, crop_size, img, \
            os.path.join(save_dir, '{:06d}.png'.format(frame_idx)),pasring_model=net, use_mask=use_mask, norm = spNorm)        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument( "--body_dir", type=str, default=None, required=True )
    parser.add_argument( "--face_dir", type=str, default=None, required=True )
    parser.add_argument( "--matrix_dir", type=str, default=None, required=True )
    parser.add_argument( "--save_dir", type=str, default=None, required=True )
    parser.add_argument( "--crop_size", type=int, default=512,)

    args = parser.parse_args()

    crop_size=args.crop_size
    if crop_size == 512:
        mode = 'ffhq'
    else:
        mode = 'None'

    save_dir = args.save_dir
    
    imgdir_inverse_align_bylist(
            body_dir=args.body_dir,
            face_dir=args.face_dir,
            matrix_dir=args.matrix_dir,
            save_dir=save_dir,      # for save dir
            crop_size=crop_size, use_mask=False)
    