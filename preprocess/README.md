<u>Preprocess are annoying and the carefully check is not achieved for the following content. I hope the following guidance works, and if any problem, please contract me.</u> 

Preprocess aims to collect training data for Make-Your-Anchor, and it can be divided into three steps:

#### Step 1: Collect your personal videos for training.

You should collect your own anchor videos for training.  In the *Make-Your-Anchor* process, you need to collect 1-5 minutes of video for training (the longer, the better). The anchor (human) should remain at the center of the frame, and the background should ideally remain static. 

Afterward, ensure the anchor stays centered in the frame, then crop and resize the video to a size of 512x512. Other resolution may works, depends on your hardware. For example:

![](assets/ref_advisor.png)

(That's my advisor with 512x512 resolution, I capture his videos for experiments.)

You finally get a video on *your_dir/raw.mp4* (The default video name is *raw.mp4* for preprocessing). Then, due to the [SHOW ](https://github.com/yhw-yhw/SHOW.git)preprocessing code I am using, which can only process 300 frames at a time, I split the video into multiple ten-second, 30fps clips. You can use the following script:

```bash
python 1_split.py --video_dir you_dir --convert_to_30fps
```

where FFMPEG is required to run this code. Then the video will be split as:
```
you_dir/
├── raw.mp4
├── raw_30fps.mp4
├── split_1/
│	├── raw.mp4
│   └── raw.wav
├── split_2/
├── split_3/
 ...
```

#### Step 2: Extract conditions.

##### Body 3D Mesh and Video Frames:

Make-Your-Anchor extracts body 3D meshes via [SHOW](https://github.com/yhw-yhw/SHOW.git). After installed this repos, run this script:

```bash
# move to the directory of SHOW
cd SHOW

# process all splits
for folder in your_dir/split_*; do
  python main.py --speaker_name -1 --all_top_dir $folder/raw.mp4
done
```

The results can be found as:


```
you_dir/
 ...
├── split_1/
│   ├── ours_exp/
│   │    ├── mica_org/
│	│	 │	 └── *png
│	│	 ...  
│	├── image/
│	│	└── *png
│	├── raw.mp4
│   └── raw.wav
 ...
```

the results in *your_dir/split_\*/ours_exp/mica_org/\*png* are required as 3D body mesh. The corresponding frames could be find in *your_dir/split_\*/image/\*png*

##### Head Conditions:

The face alignment and face parsing code are utilized to get head conditions. You can follow the same installation in README for head condition codes. We extract head meshes by the following script:

```bash
# process all splits
for folder in your_dir/split_*; do
  python main.py --speaker_name -1 --all_top_dir $folder/raw.mp4
  # face alignment
  python face_alignment.py \
     --imgdir_pth $folder/ours_exp/mica_org/ \
     --raw_imgdir_pth $folder/image/ \
     --results_dir $folder/head/ \
     --crop_size 512
  # get face mask
  python get_mask.py \
    --input_pth $folder/head/raw_aligned \
    --mask_pth $folder/head/raw_aligned_mask
done

```

#### Step 3: Organize the json files.

We use two *json* files to organize the training dataset, one for body and one for head. Run the following script:
```bash
python 3_get_body_json.py --root your_dir/  --json_file your_dir/body_train.json
python 3_get_head_json.py --json_inp_file your_dir/body_train.json  --json_file your_dir/head_train.json
```

If finished, the *your_dir/body_train.json* and *your_dir/head_train.json* are final config files for training. Then, you can follow the guidance in ***Fine-Tuning***!