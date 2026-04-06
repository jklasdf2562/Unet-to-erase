import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision.transforms import functional as F
from tqdm import tqdm


def collect_data_pair(data_dirs):
    pairs=[]
    for root in data_dirs:
        input_dir=os.path.join(root,'dataset','input')
        output_dir=os.path.join(root,'dataset','output')
        if not os.path.isdir(input_dir):
            print(f"Warning: input dir not found: {input_dir}")
            continue
        input_images=os.listdir(input_dir)
        for img in input_images:
            input_path=os.path.join(input_dir,img)
            output_path=os.path.join(output_dir,img)
            if os.path.exists(output_path):
                pairs.append((input_path,output_path))
            else:
                print(f"Warning: Output image not found for {input_path}")
    return pairs


class ImageDataset(Dataset):
    def __init__(self, data_pairs, input_size=512, mean=None, std=None, is_train=True):
        self.data_pairs = data_pairs
        self.input_size = input_size
        self.mean = mean
        self.std = std
        self.is_train = is_train

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        input_path, output_path = self.data_pairs[idx]
        input_img = Image.open(input_path).convert('RGB')
        output_img = Image.open(output_path).convert('RGB')

        input_img = F.resize(input_img, (self.input_size, self.input_size))
        output_img = F.resize(output_img, (self.input_size, self.input_size))

        if self.is_train:
            if random.random() < 0.5:
                input_img = F.hflip(input_img)
                output_img = F.hflip(output_img)
            if random.random() < 0.5:
                input_img = F.vflip(input_img)
                output_img = F.vflip(output_img)
            if random.random() < 0.5:
                angle=random.uniform(-5, 5)
                input_img = F.rotate(input_img, angle, expand=False, fill=0)
                output_img = F.rotate(output_img, angle, expand=False, fill=0)

            if random.random()<0.3:
                brightness_factor = random.uniform(0.9, 1.1)
                input_img = F.adjust_brightness(input_img, brightness_factor)
            if random.random()<0.3:
                contrast_factor = random.uniform(0.9, 1.1)
                input_img = F.adjust_contrast(input_img, contrast_factor)

        input_tensor = F.to_tensor(input_img)
        output_tensor = F.to_tensor(output_img)
        if self.is_train and random.random() < 0.2:
            noise = torch.randn_like(input_tensor) * 0.02
            input_tensor = input_tensor + noise
            input_tensor = torch.clamp(input_tensor, 0.0, 1.0)

        if self.mean is not None and self.std is not None:
            input_tensor = F.normalize(input_tensor, mean=self.mean, std=self.std)

        return input_tensor, output_tensor


def compute_mean_std(data_pairs):
    channel_sum=np.zeros(3)
    channel_squared_sum=np.zeros(3)
    num_pixels=0
    for input_path,output_path in tqdm(data_pairs,desc="Computing mean and std"):
        image=np.array(Image.open(input_path).convert('RGB'))/255.0
        channel_sum+=image.sum(axis=(0,1))
        channel_squared_sum+=(image**2).sum(axis=(0,1))
        num_pixels+=image.shape[0]*image.shape[1]

    mean_input=channel_sum/num_pixels
    std_input=np.sqrt((channel_squared_sum/num_pixels-mean_input**2))

    print(f"Computed mean: {mean_input}, std: {std_input}")
    return mean_input.tolist(),std_input.tolist()
