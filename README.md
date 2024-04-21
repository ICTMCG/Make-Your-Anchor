# Make-Your-Anchor: A Diffusion-based 2D Avatar Generation Framework

![](assets/teaser.png)

<a href='https://arxiv.org/abs/2403.16510'><img src='https://img.shields.io/badge/ArXiv-2303.09535-red'></a> 

Will appear at CVPR 2024!

## TL; DR
Make-Your-Anchor is a personalized 2d avatar generation framework based on diffusion model,  which is capable of generating realistic human videos with SMPL-X sequences as condition.

## Abstract
Despite the remarkable process of talking-head-based avatar-creating solutions, directly generating anchor-style videos with full-body motions remains challenging. In this study, we propose **Make-Your-Anchor**, a novel system necessitating only a one-minute video clip of an individual for training, subsequently enabling the automatic generation of anchor-style videos with precise torso and hand movements. Specifically, we finetune a proposed structure-guided diffusion model on input video to render 3D mesh conditions into human appearances. We adopt a two-stage training strategy for the diffusion model, effectively binding movements with specific appearances.  To produce arbitrary long temporal video, we extend the 2D U-Net in the frame-wise diffusion model to a 3D style without additional training cost, and a simple yet effective batch-overlapped temporal denoising module is proposed to bypass the constraints on video length during inference. Finally, a novel identity-specific face enhancement module is introduced to improve the visual quality of facial regions in the output videos. Comparative experiments demonstrate the effectiveness and superiority of the system in terms of visual quality, temporal coherence, and identity preservation, outperforming SOTA diffusion/non-diffusion methods. 

## Pipeline
![](assets/pipeline.png)

## Notes
- The self-collected data described in the paper will not be released due to the privacy, while we release the model trained with open dataset.
- As a person-specific approach, we plan to release the pre-trained weight from pre-training stage, and the fine-tuning code. The guidance and code for preprocess training data will be updated.
- Due to the limitation of current training dataset, our method performs better when the driven motion is in a similar style as the target person (as cross-person result shows). We plan to increase the quantity of pre-training and fine-tuning data to overcome this limitation.

## Changelog
- __[2024.04.22]__: Release the inference code and pretrained weights.

## TODO
- [x] Inference code and checkpoints
- [ ] Preprocess code and guidance
- [ ] Fine-tuning code and pre-trained weights

## Getting Started

### Environment

Our code is based on [PyTorch](https://pytorch.org/) and [Diffusers](https://huggingface.co/docs/diffusers/index). Recommended requirements can be installed via

```shell
pip install -r requirements.txt
```

To process videos, [FFmpeg](https://ffmpeg.org//) is required to be installed.

For face alignment, please download and unzip the relative files from [this link](https://onedrive.live.com/?authkey=%21ADJ0aAOSsc90neY&cid=4A83B6B633B029CC&id=4A83B6B633B029CC%215837&parId=4A83B6B633B029CC%215834&action=locate) to the folder *.\inference\insightface_func\models\\*.

### Download Inference Checkpoints

Please download the checkpoints from [Google Drive](https://drive.google.com/drive/folders/1NyEc001rdkYIIGP8TR9RAQp4Lw3UKmdh?usp=sharing), and place them in the folder *./inference/checkpoints*. Currently, we upload the checkpoints trained from open-dataset.

## Inference

We provide the inference code with our released checkpoints. After download/fine-tuned the checkpoints and place them in the *./inference/checkpoints*, the inference can be run as:

```shell
bash inference.sh
```

Specifically, five parameters should be filled with your configuration in the *inference.sh*:

```shell
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
```

After generation (it takes about 5 minutes), the results are listed in the *./inference/samples/output*.



## Video Results
### Comparisons
https://github.com/ICTMCG/Make-Your-Anchor/assets/11772240/c72910fb-2eb5-4796-8064-7abfc5f1170f

https://github.com/ICTMCG/Make-Your-Anchor/assets/11772240/2cedf65e-e311-4b04-967a-cede8ed0a75f

### Audio-Driven Results
https://github.com/ICTMCG/Make-Your-Anchor/assets/11772240/89871467-a835-48b5-a43c-8b34efbe4c0b

### Ablations
https://github.com/ICTMCG/Make-Your-Anchor/assets/11772240/64a3f48f-4e94-40ad-816d-0eafadf1adce

### Cross-person Results
https://github.com/ICTMCG/Make-Your-Anchor/assets/11772240/b688e510-f95d-44c5-80f9-cfd774134ec7

### Full-body Results
https://github.com/ICTMCG/Make-Your-Anchor/assets/11772240/644fc444-20db-46d7-a513-cd4cdb2067ba

## Citation

```BibTeX
@article{huang2024makeyouranchor,
  title={Make-Your-Anchor: A Diffusion-based 2D Avatar Generation Framework},
  author={Huang, Ziyao and Tang, Fan and Zhang, Yong and Cun, Xiaodong and Cao, Juan and Li, Jintao and Lee, Tong-Yee},
  journal={arXiv preprint arXiv:2403.16510},
  year={2024}
}
```

## Acknowledgements

Here are some great resources we benefit:

- [TalkSHOW](https://github.com/yhw-yhw/TalkSHOW) for preprocess and audio-driven inference

- [SimSwap](https://github.com/neuralchen/SimSwap.git) for the code of face preprocess

- [ControlVideo](https://github.com/YBYBZhang/ControlVideo.git) for the implementation of  full-frame attention.
