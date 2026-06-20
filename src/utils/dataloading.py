import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class ClipDataset(Dataset):
    def __init__(self, files_list, transform):
        self.files = files_list
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, key):
        sample = self.files[key]
        img = Image.open(sample["frames"])

        frames = self.transform(img)
        person = torch.tensor(sample["person"], dtype=torch.float32)
        age    = torch.tensor(sample["age"], dtype=torch.float32)

        return frames, (person, age)
    
class ImageToClip:
    def __init__(self, transform, n=8):
        self.transform = transform
        self.n = n

    def __call__(self, imgTensor):
        return torch.stack([self.transform(imgTensor) for _ in range(self.n)])