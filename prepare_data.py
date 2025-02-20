import random
import os
import numpy as np
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from tqdm import tqdm
from pathlib import Path
import json


prompts = [
    "You are a professional photographer. Give a technical description of the main colors in this image and the overall color tone of the scene.",
    "You are a renouned artist. Give a description of the colors in this image.",
    "Describe the color balance in this image and the main contrasting colors.",
    "Describe the saturation and hue of this image, focusing on colors.",
    "You have to describe this image to a colorblind person, who only sees black and white. Emphasize colors in the image."
    # "You are an artist. Describe the main colors of this image and their temperature.",
    # "As a master colorist, analyze the dominant hues of the key elements in this image and describe the overall color palette of the composition.",
    # "Imagine you're a color consultant for film. Detail the chromatic aspects of the primary subjects and the general color atmosphere of this scene.",
    # "With your expertise in digital imaging, provide a breakdown of the main color values present in the image and the overarching tonal characteristics.",
    # "Channeling your inner art critic, elaborate on the color scheme of the principal components and the holistic chromatic impression of this visual.",
    # "From the perspective of a color theorist, describe the hues of the focal points in this image and the overall color temperature of the piece.",
    # "As a seasoned cinematographer, detail the color grading of the central objects and the prevailing color mood of this frame.",
    # "With your background in visual arts, analyze the pigmentation of the key subjects and the general color ambiance of this composition.",
    # "Wearing the hat of a color psychologist, describe the primary colors of the main elements and the overall chromatic energy of this scene.",
    # "As an expert in digital restoration, provide a technical assessment of the colors of the principal objects and the global color tone of this image.",
    # "Drawing from your experience as a fine art painter, characterize the palette used for the central figures and the overarching color harmony of the entire piece.",
    # "Provide a precise and detailed description of all colors present in this image, focusing on accuracy and specificity.",
    # "Analyze this image and list every distinct color you can identify, using standard color names and any relevant color codes if possible.",
    # "Examine this image closely and describe its color composition, noting both dominant and subtle hues with as much precision as you can.",
    # "Break down the color palette of this image, identifying primary, secondary, and tertiary colors, along with any notable shades or tints.",
    # "Conduct a thorough color analysis of this image, detailing the exact hues, saturations, and values you observe throughout the composition.",
    # "Describe the full spectrum of colors in this image, from the most prominent to the least noticeable, using precise color terminology.",
    # "Perform a meticulous examination of this image's colors, reporting on both the overall color scheme and specific color details within key elements.",
    # "Offer a comprehensive color inventory of this image, noting the range and variety of hues, their relationships, and any color patterns or gradients.",
    # "Present an accurate chromatic breakdown of this image, identifying all colors and their variations, including any subtle color transitions or blends.",
    # "Deliver a precise color report for this image, detailing the full range of hues, their distribution, and any notable color interactions or contrasts.",

]

model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b", torch_dtype=torch.float16)
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b", torch_dtype=torch.float16)

model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

output_path = "data/colorization/train.json"
processed = []
with open(output_path, 'rt') as f:
    for line in f:
        processed.append(json.loads(line)['image'])


def channel_variance(img_arr):
    r, g, b = img_arr[:,:,0], img_arr[:,:,1], img_arr[:,:,2]
    var_rg = (r - g).var() 
    var_gb = (g - b).var() 
    var_rb = (r - b).var()
    return np.array([var_rg, var_gb, var_rb]).mean()

def is_valid_image(file_path):
    return Path(file_path).suffix in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]

def blip_generate(prompt, image):
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            num_beams=3,
            max_length=128,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
        )
        return processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

train_dir = 'data/colorization'
# folder_path = f"{train_dir}/flickr30k-images"
folder_path = f"{train_dir}/coco"
# folder_path = f"{train_dir}/VG_100K"
for img_path in tqdm(os.listdir(folder_path)):
    if not is_valid_image(img_path):
        print(f"Image file path not valid: {folder_path}/{img_path}")
        continue
    if Path(folder_path, img_path).as_posix() in processed:
        continue
    try:
        img = Image.open(f"{folder_path}/{img_path}").convert("RGB")
        img_arr = np.array(img)
        if channel_variance(img_arr) > 12:
            # sampled_prompts = random.sample(prompts, k=3)
            captions = []
            with open(output_path, "a") as f:
                f.write(f'{{"image": "{folder_path}/{img_path}"')
                prompt = random.choice(prompts)
                color_caption = blip_generate(prompt, img)
                color_caption = color_caption.replace("\"", "'")
                f.write(f', "prompt": "{color_caption}"')
                # for i, prompt in enumerate(sampled_prompts):
                #     color_caption = blip_generate(prompt, img)
                f.write('}\n')
        else:
            print(f"Skipping image: {folder_path}/{img_path}, it is already black and white")
            with open("bnw.txt", "a") as o:
                o.write(f"{folder_path}/{img_path}" + "\n")
    except Exception as e:
        print(f"Error processing image: {folder_path}/{img_path}")
        print("Exception message:", e)
        with open("error.txt", "a") as o:
            o.write(f"{folder_path}/{img_path}" + "\n")