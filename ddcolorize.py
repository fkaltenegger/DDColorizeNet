import random
import cv2
import einops
import numpy as np
import torch
from pytorch_lightning import seed_everything
from utils.data import HWC3, apply_color, resize_image
from utils.ddim import DDIMSampler
from utils.model import create_model, load_state_dict
from annotator.hed import HEDdetector
import torch.nn.functional as F
import sys
import os
import time



_global_model = None
_global_ddim_sampler = None

def load_global_model(model_config_path, model_weights_path):
    global _global_model, _global_ddim_sampler
    if _global_model is None or _global_ddim_sampler is None:
        model = create_model(model_config_path).cpu()
        model.load_state_dict(load_state_dict(model_weights_path, location='cuda'))
        model = model.cuda()
        _global_model = model
        _global_ddim_sampler = DDIMSampler(model)
    return _global_model, _global_ddim_sampler


def save_results(mask, colored_result, output_folder, output_name, colormask):
    
    mask_path = f"{output_folder}/mask_{output_name}.png"
    print(mask_path)
    cv2.imwrite(mask_path, cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
    
    colormask_path = f"{output_folder}/colormask_{output_name}.png"
    cv2.imwrite(colormask_path, cv2.cvtColor(colormask, cv2.COLOR_RGB2BGR))

    colorimage_path = f"{output_folder}/{output_name}.png"
    cv2.imwrite(colorimage_path, cv2.cvtColor(colored_result, cv2.COLOR_RGB2BGR))

def colorize_image(
    input_image=None,
    input_image_path=None,
    input_mask=None,
    input_mask_path=None,
    invert_mask=False,
    
    model_config_path=None,
    model_weights_path=None,
    output_folder=None,
    resolution=512,
    seed=random.randint(0, 65535),
    prompt="Colorize this image",
    n_prompt="",
    ddim_steps=20,
    scale=9.0,
    strength=1.0,
    eta=0.0,
    guess_mode=False,
    
    save_images=False,
    image_name="",
    lvl="1"
):
    model, ddim_sampler = load_global_model(model_config_path, model_weights_path)
    seed_everything(seed)
        
    output_name = f"{image_name}_lvl{lvl}_{seed}"
    
    if input_image is None:
        if input_image_path is None:
            raise ValueError("input_image or input_image_path must be provided")
        input_image = cv2.imread(input_image_path)
        input_image = HWC3(input_image)
        input_image = resize_image(input_image, resolution)
    H, W, C = input_image.shape
    
    if input_mask is None:
        if input_mask_path is None:
            print("No mask provided, using HED to generate mask")
            input_mask = apply_hed(input_image)
        else:
            input_mask = cv2.imread(input_mask_path)
        input_mask = HWC3(input_mask)
        input_mask = resize_image(input_mask, resolution)
    
    if invert_mask:
        input_mask = cv2.bitwise_not(np.uint8(input_mask))
        
    mask_save = input_mask.copy()
        
    
    num_samples = 1
    control = torch.from_numpy(input_image.copy()).float().cuda() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()
    
    input_mask = torch.from_numpy(input_mask.copy()).float().cuda() / 255.0
    input_mask = torch.stack([input_mask for _ in range(num_samples)], dim=0)
    input_mask = einops.rearrange(input_mask, 'b h w c -> b c h w').clone()
    
    input_mask = F.interpolate(input_mask, size=(H // 8, W // 8), mode='bilinear', align_corners=False)
    if input_mask.shape[1] == 3:
        alpha_channel = torch.zeros_like(input_mask[:, :1, :, :])
        input_mask = torch.cat([input_mask, alpha_channel], dim=1)
    
    x0 = model.encode_first_stage(control).mean
    x0 = F.interpolate(x0, size=(H // 8, W // 8), mode='bilinear', align_corners=False)
    
    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt] * num_samples)]}
    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
    shape = (4, H // 8, W // 8)
    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)

    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples, shape, cond, verbose=False, eta=eta,
                                                 unconditional_guidance_scale=scale, unconditional_conditioning=un_cond,
                                                 mask=input_mask, x0=x0)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    results = [x_samples[i] for i in range(num_samples)]
    
    colored_results = [apply_color(input_image, result) for result in results]


    if save_images:

        save_results(mask=mask_save, 
                     colored_result=colored_results[0], 
                     output_folder=output_folder, 
                     output_name=output_name,
                     colormask=results[0])

    return input_mask, colored_results[0]

    


img_nr = sys.argv[1]

seed = random.randint(0, 65535)
apply_hed = HEDdetector()

output_path = f"output_ddcolorizenet/"
os.makedirs(output_path, exist_ok=True)
    
mask1, image1 = colorize_image(
    model_config_path="./models/cldm_v21.yaml",
    model_weights_path="./models/colorizenet-sd21.ckpt",
    seed=seed,
    
    input_image_path=f"test_imgs/ddcolorizenet/{img_nr}.png",
    input_mask_path=f"test_imgs/cats/{img_nr}.png",
    output_folder=output_path,
    save_images=True,
    lvl="1",
    image_name=img_nr,
    invert_mask=False,
    ddim_steps=20,
)

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

mask2, image2 = colorize_image(
    model_config_path="./models/cldm_v21.yaml",
    model_weights_path="./models/colorizenet-sd21.ckpt",
    seed=seed,
    
    input_image=image1,
    input_mask_path=f"test_imgs/cats/{img_nr}.png",
    output_folder=output_path,
    save_images=True,
    lvl="2",
    image_name=img_nr,
    invert_mask=True,
    ddim_steps=20,
)

    
    
    
    