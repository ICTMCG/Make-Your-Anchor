import os
import json
from tqdm import tqdm
import random

import argparse

parser = argparse.ArgumentParser()
parser.add_argument( "--json_inp_file", type=str, default=None, required=True)
parser.add_argument( "--json_pth", type=str, default=None, required=True)
args = parser.parse_args()

data = []
with open(args.json_inp_file, 'rt') as f_r:
    for line in f_r:
        data.append(json.loads(line))


with open(args.json_pth, 'w') as f:
    for d in tqdm(data):
        src = d['source']
        tgt = d['target']
        ref = d['reference']

        image_name = os.path.basename(src)
        root_dir = os.path.dirname(tgt)
        root_dir = os.path.join(root_dir[:-5], 'head_crop')
        src = os.path.join(root_dir, 'aligned', image_name[:-4]+'.jpg')
        tgt = os.path.join(root_dir, 'raw_aligned', image_name[:-4]+'.png')
        mask_pth = os.path.join(root_dir, 'raw_aligned_mask', image_name[:-4]+'.png')

        ref_name = os.path.basename(ref)
        ref = os.path.join(root_dir, 'raw_aligned', ref_name[:-4]+'.png')

        if not os.path.exists(mask_pth):
            print(mask_pth+' mask')
            continue
        if not os.path.exists(ref):
            print('ref:'+ref)
            continue
        if not os.path.exists(src):
            print(src + ' src')
            continue
        if not os.path.exists(tgt):
            print(tgt + ' tgt')
            continue

        one_item = json.dumps({'source':src, 'target':tgt, 'reference':ref, 'mask':mask_pth,})
        f.write(one_item)
        f.write('\n')