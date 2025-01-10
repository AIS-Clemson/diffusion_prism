# Diffusion Prism: Enhancing Diversity and Morphology Consistency in Mask-to-Image Diffusion

News: This paper is accepted by the [WACV 2024 4th Workshop on Image/Video/Audio Quality in Computer Vision and Generative AI](https://wacv2025-image-quality-workshop2.github.io/index.html)


## Introduction

Diffusion Prism is a training-free framework that efficiently transforms binary masks into realistic and diverse samples while preserving morphological features. We explored that a small amount of artificial noise will significantly assist the image-denoising process. To prove this novel mask-to-image concept, we use nano-dendritic patterns as an example to demonstrate the merit of our method compared to existing controllable diffusion models. We also extend the proposed framework to other biological patterns, highlighting its potential applications across various fields. 


## Quick Tutorial
1. First, please download stable-diffusion-v1-5 model file from: [https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt) and place it into the `diffusion_prism\models\ldm\stable-diffusion-v1\` folder. <br>

2. Run `Flame_diffuser_perlin_mask.py` as a demo to show the proposed 'perlin_mask' method from the paper.

3. Go to [exp](exp) to check the mask-related functions such as `mask_generator.py` <br>
Run `dataset_eval.py` to produce the results of FID, CLIP Score, and CLIP Confidence. It will generate annotation for the test folder as well.

We will update more details later according to the request. Please contact us anytime if you have questions.

## Sample Dataset

<img src="./Figure/teaser_2.jpg" width=70%>

- **Dataset:** Download from [Google Drive]()

---


## Key Features

- **Training-Free Diffusion Framework:** Generates images from binary skeletons without the need for model training or fine-tuning.
- **Diverse Backgrounds:** Creates images with varied and realistic backgrounds, enhancing model generalizability.

## Methodology

<img src="./Figure/latent_4.jpg" width=100%>

**Diffusion Process:**
   - Combines masks with controllable noise, processed through a Variational Autoencoder (VAE) to generate latent variables.
   - The denoising U-Net refines these variables to produce realistic images guided by text prompts.


## Experimental Results

<img src="./Figure/grid_3.jpg" width=100%>

- **High-Quality:** Lowest FID score compared to other methods, indicating better realistic styles.
- **Consistency:** Morphology preserving, the skeleton shape is well-kept in synthesized images.

<img src="./Figure/abl.png" width=70%>

For more details, visit the [Project Page](https://arazi2.github.io/aisends.github.io/project/Prism).



# Citation 
<a href="https://arxiv.org/abs/2501.00944">Diffusion Prism: Enhancing Diversity and Morphology Consistency in Mask-to-Image Diffusion

@article{wang2025diffusion,
  title={Diffusion Prism: Enhancing Diversity and Morphology Consistency in Mask-to-Image Diffusion},
  author={Wang, Hao and Chen, Xiwen and Bastola, Ashish and Qin, Jiayou and Razi, Abolfazl},
  journal={arXiv preprint arXiv:2501.00944},
  year={2025}
}
