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

        with Image.open(sample["frames"]) as img:
            img = img.convert("RGB")
            frames = self.transform(img)

        #frames = self.transform(img)
        person = torch.tensor(sample["person"], dtype=torch.float32)
        age    = torch.tensor(sample["age"], dtype=torch.float32)

        return frames, (person, age)
    
class ImageToClip:
    def __init__(self, baseTransform, augmentTransform, n=8):
        self.base = baseTransform
        self.augment = augmentTransform
        self.n = n

    def __call__(self, img):
        img = self.base(img)
        return torch.stack([self.augment(img) for _ in range(self.n)])