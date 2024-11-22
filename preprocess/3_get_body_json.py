import os
import json
from tqdm import tqdm
import random
from glob import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument( "--root", type=str, default=None, required=True)
parser.add_argument( "--json_file", type=str, default=None, required=True)
args = parser.parse_args()

temp_dir_paths = glob(args.root+'/split_*')

# filter the bad dirs
dir_paths = []
for dir_path in temp_dir_paths:
    src_dir = os.path.join(dir_path, 'ours_exp/mica_org/000001.png')
    if os.path.exists(src_dir):
        dir_paths.append(dir_path)

def get_body_json(json_pth, dir_paths):
    with open(json_pth, 'w') as f:
        for dir_path in tqdm(dir_paths):
            tgt_dir = os.path.join(dir_path, 'image')
            src_dir = os.path.join(dir_path, 'ours_exp/mica_org')
            img_names = os.listdir(tgt_dir)
            for img_name in img_names:
                ref_img_name = random.choice(img_names)
                ref_img_pth = os.path.join(tgt_dir, ref_img_name)

                tgt_pth = os.path.join(tgt_dir, img_name)
                src_pth = os.path.join(src_dir, img_name[:-4] + '.png')
                if not os.path.exists(tgt_pth):
                    continue
                if not os.path.exists(src_pth):
                    continue
                data = json.dumps({'source':src_pth, 'target':tgt_pth, 'reference':ref_img_pth})
                f.write(data)
                f.write('\n')


get_body_json(args.json_file, dir_paths)
