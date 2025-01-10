"""make variations of input image"""

import os
import cv2
import random
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext

import time
import copy
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms


from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def normalize_image(image):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    normalized_image = (image / 1.0 - mean) / std
    return normalized_image


def add_gaussian_noise(tensor, mean, std):
    noise = torch.randn_like(torch.Tensor(np.array(tensor))) * std + mean
    return tensor + noise.numpy()






def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    print(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

# path = opt.init_img
def load_img(image_name, path, mask, noise_std):
    # style = Image.open('./train/style.png').convert("RGB")
    image = Image.open(path).convert("RGB")
    # image = image.convert("RGB")

    

    
    
    # z = len(image.mode)
    w, h = mask.size
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32
    ratio = w / h
    if w > h:
        w = 512
        h = int(w / ratio)
        if h % 64 != 0:
            h = int((h // 64 + 1) * 64)
    else:
        h = 512
        w = int(h * ratio)
        if w % 64 != 0:
            w = int((w // 64 + 1) * 64)
    print(f"loaded input image from {path}, resize to ({w}, {h}) ")
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    mask = mask.resize((w, h), resample=PIL.Image.LANCZOS)


    # style = np.array(style)
    image = np.array(image)
    
    mask = np.array(mask)#.astype(np.float32) / 255.0
    
    # image = image * mask
    # image = image.astype(np.uint8)
    
    
    
    
    # kernel_size = 7
    # fire_thred = 0.9
    # # zone_overlap_thred = 0.9
    # # flame_thred = 0.1
    # # num_areas = 100
    
    # # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = mask
    # # w,h = gray.shape
    # # ax2 = plt.subplot(2,2,2)
    # # ax2.imshow(gray)
    # # ax1.colorbar()
    # # gray = cv2.blur(gray, (kernel_size,kernel_size))
    # # image_IR =  cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # # print(np.mean(gray))
    # # print(np.median(gray))
    # # print(np.max(gray))
    
    # thred = np.mean(gray) + np.median(gray)

    # # np.sort(gray.reshape(-1))
    # gray_sort = np.sort(gray.reshape(-1))
     

    # # X2 = gray_sort
    # # F2 = np.array(range(len(X2)))/float(len(X2))
    # # plt.plot(X2, F2)
    
    # # plt.scatter(np.arange(len(gray.reshape(-1))),np.sort(gray.reshape(-1)))
    # seg = copy.deepcopy(gray)
    # seg[seg < fire_line] = 0
    # # plt.imshow(seg)
    # seg_blur = cv2.blur(seg, (kernel_size,kernel_size))
    
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))

    # mask = cv2.morphologyEx(seg_blur, cv2.MORPH_OPEN, kernel, iterations = 1)



    # plt.imshow(seg_blur)
    mask =  cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    mask_out = copy.deepcopy(cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))

    
    
    # mask_neg = (255-mask)/255
    # plt.imshow(mask_neg)
    # mask_neg =  cv2.cvtColor(mask_neg, cv2.COLOR_GRAY2RGB)

    # mask =  cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # plt.imshow(mask)



    
    # if random.random() > 0.5:
    #     image = 255-image
        
    # # Generate random factors for each layer
    # random_factors = np.random.uniform(0, 1, size=3)
    # # random_factors = [1, 0.5, 0.25]
    # for i in range(3):
    #     image[:, :, i] = image[:, :, i] * random_factors[i]
    
    # image = copy.deepcopy(mask)

    # Map color
    mask[:, :, 0] = mask[:, :, 0] * np.random.uniform(0.7, 0.9)
    mask[:, :, 1] = mask[:, :, 1] * np.random.uniform(0.2, 0.4)
    mask[:, :, 2] = mask[:, :, 2] * np.random.uniform(0, 0.1)
    
    # mask[mask==0] = 1
    # mask = (mask - np.min(mask))/(np.max(mask)-np.min(mask)) *255
    
    
    # Add noise
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = mask + np.random.normal(0, noise_std, size=(w,h,3))
    mask = (mask - np.min(mask))/(np.max(mask)-np.min(mask)) *255
    # mask = mask*255
    mask = mask.astype(np.uint8)
    

    # # image = 0.5*np.array(style) + 0.5*np.array(image)


    # image = image * mask_neg
    # image[mask>0] = 0.5*image[mask>0] + 0.5*mask[mask>0]
    image = 1*image.astype(int) + np.random.uniform(0.6, 0.7)*mask.astype(int)

    image[image>255] = 255
    image = image.astype(np.uint8)
    # plt.imshow(image)



    
    # To tensor
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    
    # mask =  cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    return (2.*image - 1.), mask_out



def print_text_on_image(image, text, font_path, font_size, text_position, text_color):
    # Open the image
    # image = Image.open(image_path)

    # Create an ImageDraw object
    draw = ImageDraw.Draw(image)

    # Specify the font and size
    font = ImageFont.truetype("arial.ttf", size=font_size)
    # text = 'test'
    # Draw the text on the image
    draw.text(text_position, text, font=font, fill=(255, 255, 255))

    # Save the modified image
    # image.save("image_with_text.jpg")
    return image
    

    
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image",
        default="test2.jpg"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2img-samples"
    )

    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )

    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save indiviual samples. For speed measurements.",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=2,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/sd-v1-1.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    opt = parser.parse_args()
    # seed_everything(opt.seed)
    
    img_path = './test_img/samples/'
    img_list = os.listdir(img_path)
    img_file = random.choice(img_list)
    image_name = img_file.split('.')[0]
    
    opt.init_img = img_path+img_file
    # # opt.init_img = "./test_img/image_skeleton_dia.png"
    # opt.prompt = "polygon, nano materials, dendrite, lab sample, chemistry, biology, encryption, code"
    opt.prompt = "A paper with lab sample is on an office desktop, close-up camera, photorealistic, highresolution, realistic, \
        office, ambient lighting, "
    opt.scale = 5
    opt.strength = 0.9
    opt.ddim_steps = 20
    opt.n_samples = 1
    opt.n_iter = 10
    opt.ckpt = "models/ldm/stable-diffusion-v1/sd-v1-1.ckpt"
    
    noise_std = 0.1
    # num_sample = 100
    # path = opt.init_img

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    

    
    if opt.plms:
        raise NotImplementedError("PLMS sampler not (yet) supported")
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1
    
        
    assert os.path.isfile(opt.init_img)
    init_image = load_img(opt.init_img, noise_std).to(device)
    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space



    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

    assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(opt.strength * opt.ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)

                        # encode (scaled latent)
                        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                        # decode it
                        samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,)

                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        if not opt.skip_save:
                            for x_sample in x_samples:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                # x_sample = normalize_image(x_sample)
                                Image.fromarray(x_sample.astype(np.uint8)).save(
                                    os.path.join(sample_path, f"{image_name}_{base_count:05}.png"))
                                base_count += 1
                        all_samples.append(x_samples)

                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    grid_count += 1

                toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")



from visualizer import get_local
get_local.activate() 


if __name__ == "__main__":
    # main()
    
    batch_generation = True
    
    
    data_label = {"image:"}
    
    if batch_generation:
        
        parser = argparse.ArgumentParser()
    
        parser.add_argument(
            "--prompt",
            type=str,
            nargs="?",
            default="a painting of a virus monster playing guitar",
            help="the prompt to render"
        )
    
        parser.add_argument(
            "--init-img",
            type=str,
            nargs="?",
            help="path to the input image",
            default="test2.jpg"
        )
    
        parser.add_argument(
            "--outdir",
            type=str,
            nargs="?",
            help="dir to write results to",
            default="outputs/img2img-samples"
        )
    
        parser.add_argument(
            "--skip_grid",
            action='store_true',
            help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
        )
    
        parser.add_argument(
            "--skip_save",
            action='store_true',
            help="do not save indiviual samples. For speed measurements.",
        )
    
        parser.add_argument(
            "--ddim_steps",
            type=int,
            default=50,
            help="number of ddim sampling steps",
        )
    
        parser.add_argument(
            "--plms",
            action='store_true',
            help="use plms sampling",
        )
        parser.add_argument(
            "--fixed_code",
            action='store_true',
            help="if enabled, uses the same starting code across all samples ",
        )
    
        parser.add_argument(
            "--ddim_eta",
            type=float,
            default=0.0,
            help="ddim eta (eta=0.0 corresponds to deterministic sampling",
        )
        parser.add_argument(
            "--n_iter",
            type=int,
            default=1,
            help="sample this often",
        )
        parser.add_argument(
            "--C",
            type=int,
            default=4,
            help="latent channels",
        )
        parser.add_argument(
            "--f",
            type=int,
            default=8,
            help="downsampling factor, most often 8 or 16",
        )
        parser.add_argument(
            "--n_samples",
            type=int,
            default=2,
            help="how many samples to produce for each given prompt. A.k.a batch size",
        )
        parser.add_argument(
            "--n_rows",
            type=int,
            default=0,
            help="rows in the grid (default: n_samples)",
        )
        parser.add_argument(
            "--scale",
            type=float,
            default=5.0,
            help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
        )
    
        parser.add_argument(
            "--strength",
            type=float,
            default=0.75,
            help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
        )
        parser.add_argument(
            "--from-file",
            type=str,
            help="if specified, load prompts from this file",
        )
        parser.add_argument(
            "--config",
            type=str,
            default="configs/stable-diffusion/v1-inference.yaml",
            help="path to config which constructs model",
        )
        parser.add_argument(
            "--ckpt",
            type=str,
            default="models/ldm/stable-diffusion-v1/sd-v1-1.ckpt",
            help="path to checkpoint of model",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="the seed (for reproducible sampling)",
        )
        parser.add_argument(
            "--precision",
            type=str,
            help="evaluate at this precision",
            choices=["full", "autocast"],
            default="autocast"
        )
        
        opt = parser.parse_args()
        # seed_everything(opt.seed)
        
        # sample_path = './test_img/samples/'
        # sample_file_list = os.listdir(sample_path)
        # sample_file = random.choice(sample_file_list)
        
        # opt.init_img = sample_path+sample_file
        # # opt.init_img = "./test_img/image_skeleton_dia.png"
        # opt.prompt = "polygon, nano materials, dendrite, lab sample, chemistry, biology, encryption, code"
        opt.prompt = " "
        # opt.scale = 100
        # opt.strength = 0.2
        # opt.ddim_steps = 100
        opt.n_samples = 1
        opt.n_iter = 1
        opt.ckpt = "models/ldm/stable-diffusion-v1/sd-v1-1.ckpt"
        opt.skip_grid = True
        opt.watermark = False
        
        num_sample = 1
        # path = opt.init_img
    
        config = OmegaConf.load(f"{opt.config}")
        model = load_model_from_config(config, f"{opt.ckpt}")
    
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)
        
        
        # if opt.plms:
        #     raise NotImplementedError("PLMS sampler not (yet) supported")
        #     sampler = PLMSSampler(model)
        # else:
        #     sampler = DDIMSampler(model)
    
        # os.makedirs(opt.outdir, exist_ok=True)
        # outpath = opt.outdir
    
        # batch_size = opt.n_samples
        # n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
        # if not opt.from_file:
        #     prompt = opt.prompt
        #     assert prompt is not None
        #     data = [batch_size * [prompt]]
        
        # else:
        #     print(f"reading prompts from {opt.from_file}")
        #     with open(opt.from_file, "r") as f:
        #         data = f.read().splitlines()
        #         data = list(chunk(data, batch_size))
        
        # sample_path = os.path.join(outpath, "samples")
        # os.makedirs(sample_path, exist_ok=True)
        # base_count = len(os.listdir(sample_path))
        # grid_count = len(os.listdir(outpath)) - 1
        
        
        # img_path = './train/samples_mask/'
        # img_path = 'D:/Data/Flame 2/254p Dataset/254p RGB Images/'
        # img_path = './wildfire_images/mask/'
        # img_path = './wildfire_images/FLAME 2/'
        img_path = './bus.jpg'


        # mask_path = 'D:/Data/Flame 2/254p Dataset/254p Thermal Images/'
        mask_path = './wildfire_images/mask/'
        mask_file = random.choice(os.listdir(mask_path))

        img_list = os.listdir(img_path)
    
        precision_scope = autocast if opt.precision == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    tic = time.time()
                    all_samples = list()
                    
                    i = 0

                    while i < num_sample:
                        try:
                            i += 1
                            img_file = random.choice(img_list)
                            # img_file = img_list[i]
                            image_name = img_file.split('.')[0]
                            
                                                        

                            mask = Image.open(mask_path+mask_file).convert("L")
                            # mask = Image.open(path).convert("L")

                        
                                    
                            opt.init_img = img_path+img_file
                            # opt.init_img = "./test_img/image_skeleton_dia.png"
                            prompt_1 = "photo realistic, high resolution, bright day, flame burning in the snow covered by the smoke"
                            # prompt_2 = "photo realistic, high resolution, bright day, flame in the desert covered by the smoke"
                            prompt_3 = "photo realistic, high resolution, bright day, flame burning in the forest covered by the smoke"
                            # prompt_4 = "photo realistic, high resolution, bright ambient light, city, flame covered by the smoke"

                            prompt_list = [prompt_1, prompt_3]
                            # opt.prompt = random.choice(prompt_list)
                            # opt.prompt = "random, blood, retina, vessel, realistic, colorful, high resolution, texture, polygon, nano materials, dendrite, chemistry, lab sample"
                            # opt.prompt = "retina vessel"
                            # opt.prompt = "photo realistic, wildfire in the forest, flame and fire in the smoke, snow, forest, high resolution"
                            # opt.prompt = 'fire burning in the smoke'
                            # opt.prompt = "photo realistic, wildfire in the forest, flame and fire in the smoke and snow, high resolution"
                            # opt.prompt = "photo realistic, cars and people in the city, city view, traffic lights, high resolution"
                            opt.prompt = 'rainy day'

                            for scale in range(0,20,2):
                                
                                for strength in np.arange(0.1, 0.9, 0.1):
                                    
                                    opt.scale = scale# random.randint(5, 15)
                                    opt.strength = strength# random.uniform(0.6, 0.7)
                                    # opt.scale = int(10/opt.strength)
                                    opt.ddim_steps = int(11/(opt.strength))
                                    # if opt.ddim_steps>=100: opt.ddim_steps=100
                                    noise_std = 0.05 # random.uniform(0.1, 0.2)
                                    
                                    if opt.plms:
                                        raise NotImplementedError("PLMS sampler not (yet) supported")
                                        sampler = PLMSSampler(model)
                                    else:
                                        sampler = DDIMSampler(model)
                                
                                    os.makedirs(opt.outdir, exist_ok=True)
                                    outpath = opt.outdir
                                
                                    batch_size = opt.n_samples
                                    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
                                    if not opt.from_file:
                                        prompt = opt.prompt
                                        assert prompt is not None
                                        data = [batch_size * [prompt]]
                                
                                    else:
                                        print(f"reading prompts from {opt.from_file}")
                                        with open(opt.from_file, "r") as f:
                                            data = f.read().splitlines()
                                            data = list(chunk(data, batch_size))
                                
                                    sample_path = os.path.join(outpath, "samples")
                                    os.makedirs(sample_path, exist_ok=True)
                                    base_count = len(os.listdir(sample_path))
                                    grid_count = len(os.listdir(outpath)) - 1
                                    
                                    
                                    assert os.path.isfile(opt.init_img)
                                    init_image, mask_out = load_img(image_name, opt.init_img, mask, noise_std)
                                    init_image = init_image.to(device)
                                    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
                                    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space
                
                                    
                
                                    assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
                                    t_enc = int(opt.strength * opt.ddim_steps)
                                    print(f"target t_enc is {t_enc} steps")
                                
                                    try:
                                        sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)
                                    except:
                                        opt.ddim_steps = 100
                                        sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)
                
                
                                    for n in trange(opt.n_iter, desc="Sampling"):
                                        
                                        for prompts in tqdm(data, desc="data"):
                                            uc = None
                                            if opt.scale != 1.0:
                                                uc = model.get_learned_conditioning(batch_size * [""])
                                            if isinstance(prompts, tuple):
                                                prompts = list(prompts)
                                            c = model.get_learned_conditioning(prompts)
                    
                                            # encode (scaled latent)
                                            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                                            # decode it
                                            samples, cc = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale,
                                                                      unconditional_conditioning=uc,)
                    
                                            x_samples = model.decode_first_stage(samples)
                                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                                            
                                            cache = get_local.cache # ->  {'your_attention_function': [attention_map]}
                    
                                            if not opt.skip_save:
                                                for x_sample in x_samples:
                                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                                
                                                    image = Image.fromarray(x_sample.astype(np.uint8))
                                                    # image = normalize_image(image)
                                                    strength = round(opt.strength,2)
                                                    noise = round(noise_std,2)
                                                    font_size = 20
                                                    
                                                    if opt.watermark==True:
                                                        image = print_text_on_image(image, f"scale={opt.scale} strength={strength} steps={opt.ddim_steps} noise={noise}", "arial.ttf", font_size, (10, 10), (255, 255, 255))
                                                        image = print_text_on_image(image, f"{opt.prompt}", "arial.ttf", font_size, (10, 30), (255, 255, 255))
                                                        image = print_text_on_image(image, f"{img_file}", "arial.ttf", font_size, (10, 50), (255, 255, 255))
                    
                                                    image.save(
                                                        os.path.join(sample_path, f"{image_name}_{scale}_{strength}_.png"))
                                                    cv2.imwrite(os.path.join(sample_path, f"{image_name}_mask.png"), mask_out)
                                                    base_count += 1
                                            all_samples.append(x_samples)
                    
                                    if not opt.skip_grid:
                                        # additionally, save as grid
                                        grid = torch.stack(all_samples, 0)
                                        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                                        grid = make_grid(grid, nrow=n_rows)
                    
                                        # to image
                                        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                                        Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                                        grid_count += 1
                    
                                    toc = time.time()
                            
                                    
                        except:
                            print('skip...')
                            
                        
            
                print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
                      f" \nEnjoy.")

            


# def norm_image(image):
#     """
#     Normalization image
#     :param image: [H,W,C]
#     :return:
#     """
#     image = image.copy()
#     image -= np.max(np.min(image), 0)
#     image /= np.max(image)
#     image *= 255.
#     return np.uint8(image)

        


# print(opt.prompt)                

# # attention_maps = cache['UNetModel.forward']
# attention_maps = cache['CrossAttention.forward']
# # attention_maps = cache['BasicTransformerBlock._forward']


# len(attention_maps)
# attention_maps[-1].shape
# channel = len(attention_maps[-1])


# # attention_maps = norm_image(attention_maps[-1])
# attention_maps = attention_maps[-1]


# # Assuming `attention_map` has shape 16x4096x77 and `token_index` is the index of "wildfire"
# attention_to_wildfire = attention_maps[:, :, 76]  # Extracting the vectors related to "wildfire", shape 16x4096
# # attention_to_wildfire = attention_maps.sum(axis=2)  # Extracting the vectors related to "wildfire", shape 16x4096


# # Reshape to get 64x64 grid for each head
# attention_to_wildfire_reshaped = attention_to_wildfire.reshape(16, 64, 64)  # shape becomes 16x64x64

# # Average across heads (optional)
# average_attention_to_wildfire = attention_to_wildfire_reshaped.mean(axis=0)  # shape becomes 64x64

# average_attention_to_wildfire = norm_image(average_attention_to_wildfire)
# average_attention_to_wildfire = cv2.resize(average_attention_to_wildfire, (512,512))

# # Now, `average_attention_to_wildfire` is your final 64x64 attention map that can be visualized.
# plt.figure(dpi=100)
# plt.imshow(image)
# plt.imshow(average_attention_to_wildfire, alpha=0.5, interpolation='nearest', cmap='jet')



        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
