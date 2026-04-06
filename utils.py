import torch
import numpy as np

def denormalize(tensor, mean, std):
    mean=torch.tensor(mean).view(1,3,1,1).to(tensor.device)
    std=torch.tensor(std).view(1,3,1,1).to(tensor.device)
    return tensor * std + mean

def tensor_to_uint8_image(tensor):
    # tensor: C,H,W in [0,1]
    arr = tensor.cpu().numpy().transpose(1,2,0)
    arr = np.clip(arr*255, 0, 255).astype(np.uint8)
    return arr
