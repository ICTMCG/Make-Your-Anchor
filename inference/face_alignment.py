import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from insightface_func.face_detect_crop_single import Face_detect_crop

import argparse

def align(img, M, crop_size):
    align_img = cv2.warpAffine(img, M, (crop_size, crop_size), borderValue=0.0)
    return align_img

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)


def imgdir_align(imgdir_path, raw_imgdir_pth,  detect_model, results_dir='./temp_results', crop_size=224):    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'aligned'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'raw_aligned'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'matrix'), exist_ok=True)
            
    for file_name in tqdm(os.listdir(imgdir_path)): 
        img_pth = os.path.join(imgdir_path, file_name)
        img = cv2.imread(img_pth)

        raw_img_pth = os.path.join(raw_imgdir_pth, file_name[:-4]+'.jpg')
        raw_file_name = file_name[:-4]+'.jpg' if os.path.exists(raw_img_pth) else file_name[:-4]+'.png'
        raw_img_pth = os.path.join(raw_imgdir_pth, raw_file_name)
        if not os.path.exists(raw_img_pth):
            continue
        raw_img = cv2.imread(raw_img_pth)

        if img is not None:
            detect_results = detect_model.get(img,crop_size)

            if detect_results is not None:

                img_align_crop_list = detect_results[0]
                img_mat_list = detect_results[1]
        
                for img_align_crop in img_align_crop_list:
                    cv2.imwrite(os.path.join(results_dir, 'aligned', file_name[:-4]+'.jpg'), img_align_crop)

                    raw_img_align_crop = align(raw_img, img_mat_list[0], crop_size)
                    cv2.imwrite(os.path.join(results_dir, 'raw_aligned', raw_file_name), raw_img_align_crop)
                    np.save(os.path.join(results_dir, 'matrix', file_name.split('.')[0]), img_mat_list[0])
                    break

            else:
                print('not detected in {}'.format(img_pth))
                if not os.path.exists(results_dir):
                    os.mkdir(results_dir)

        else:
            pass





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument( "--imgdir_pth", type=str, default=None, required=True )
    parser.add_argument( "--raw_imgdir_pth", type=str, default=None, required=True )
    parser.add_argument( "--results_dir", type=str, default=None, required=True )
    parser.add_argument( "--crop_size", type=int, default=512 )

    args = parser.parse_args()


    imgdir_pth = args.imgdir_pth
    raw_imgdir_pth = args.raw_imgdir_pth

    crop_size=args.crop_size
    if crop_size == 512:
        mode = 'ffhq'
    else:
        mode = 'None'
    
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640),mode=mode)

    results_dir = args.results_dir
    
    imgdir_align(imgdir_pth, raw_imgdir_pth, app, crop_size=crop_size, results_dir=results_dir)