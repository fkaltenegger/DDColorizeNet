import json
import random

import cv2
import numpy as np

from PIL import Image
from skimage.segmentation import slic
from torch.utils.data import Dataset
from torchvision import transforms as T

#TODO Test if everything works fine

def make_slic_mask(image, n_segments=100, compactness=20, sigma=1):
    segments = slic(image, n_segments=n_segments, compactness=compactness, sigma=sigma)
    segment_ids = np.unique(segments)

    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_img)

    most_uniform_segments = get_most_uniform_segments(h, segments, segment_ids)
    most_colorful_segments = get_most_saturated_segments(s, segments, most_uniform_segments)
    k = random.randint(1, 3)
    high_sat_sids = random.sample(most_colorful_segments, k=k)

    combined_mask = np.zeros_like(segments, dtype=np.uint8)

    hint_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hint = cv2.cvtColor(hint_gray, cv2.COLOR_GRAY2RGB)
    for sid in high_sat_sids:
        binary_mask = np.zeros_like(segments, dtype=np.uint8)
        binary_mask[np.isin(segments, sid)] = 1
        combined_mask += binary_mask

        color = get_patch_color(image, binary_mask)
        hint = color_patch(hint, binary_mask, color)
    return hint, binary_mask

def to_pil(x, normalize=False):
    if normalize:
        x = x * 0.5 + 0.5
    x = x * 255
    x = x.astype(np.uint8)
    return x

def get_most_uniform_segments(hue, segments, segment_ids):
    seg_color_var = []
    for sid in segment_ids:
        segment_hue = hue[segments == sid]
        seg_color_var.append((sid, segment_hue.var()))

    seg_color_var.sort(key=lambda x: x[1])
    return seg_color_var[:6]

def get_most_saturated_segments(saturation, segments, segment_ids):
    seg_sats = []
    for sid, _ in segment_ids:
        segment_sat = saturation[segments == sid]
        seg_sats.append((sid, segment_sat.mean()))

    seg_sats.sort(key=lambda x: x[1], reverse=True)
    high_sat_segments = [seg_sats[i][0] for i in range(3)]
    return high_sat_segments[:3]

def get_patch_color(image, binary_mask):
    def get_mean_color(ch):
        ch = int((ch * binary_mask).sum() // binary_mask.sum())
        ch = np.clip(ch, 0, 255)
        return ch

    r,g,b = np.array_split(image, 3, axis=2)
    r = get_mean_color(r.squeeze())
    g = get_mean_color(g.squeeze())
    b = get_mean_color(b.squeeze())

    return r,g,b

def color_patch(image, binary_mask, color):
    mask = binary_mask.astype(bool)

    r,g,b = color

    image[mask, 0] = r
    image[mask, 1] = g
    image[mask, 2] = b
    return image

class ColorizationDataset(Dataset):
    def __init__(self, data_root):
        self.data = []
        self.data_root = data_root
        with open(f'{self.data_root}train.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]

        image_path = item['image']
        prompt = item['prompt']

        target = cv2.imread(image_path)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
        L, a, b = np.split(target_lab, 3, axis=2)
        source = L # Take only L channel from target

        # Do not forget that OpenCV read images in BGR order.
        # source = cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        if random.random() < 0.2:
            color_jitter = T.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.6, hue=0.4)
            target = color_jitter(Image.fromarray(target))
            target = np.array(target)

        # Get mask for stroke simulation using SLIC
        hint, mask = make_slic_mask(target, n_segments=15, compactness=20, sigma=1)

        source = cv2.resize(source, (512, 512), interpolation=cv2.INTER_CUBIC)
        source = source[..., None]
        target = cv2.resize(target, (512, 512), interpolation=cv2.INTER_CUBIC)
        hint = cv2.resize(hint, (512, 512), interpolation=cv2.INTER_CUBIC)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0
        hint = hint.astype(np.float32) / 255.0
        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        # return dict(jpg=target, txt=prompt, hint=np.concatenate([source, hint], axis=3))
        return dict(jpg=target, txt=prompt, hint=hint, source=source, mask=mask)

if __name__ == "__main__":
    dataset = ColorizationDataset("data/colorization/")
    print("Total items: ", len(dataset))
    sample = dataset[random.choice(list(range(len(dataset))))]
    Image.fromarray(to_pil(sample["jpg"], normalize=True)).save("target.png")
    Image.fromarray(to_pil(sample["source"])).save("source.png")
    Image.fromarray(to_pil(sample["hint"])).save("hint.png")