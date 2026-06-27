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
    
class FeaturesDataset(Dataset):
    def __init__(self, shard_dir, split):
        self.shard_dir = shard_dir
        self.split     = split
 
        index = torch.load(os.path.join(shard_dir, f"{split}_index.pt"), weights_only=True)

        self.shard_ids = index["shard_ids"]     # (N,)
        self.local_ids = index["local_ids"]     # (N,)
        self.persons   = index["persons"]       # (N,)
        self.ages      = index["ages"]          # (N,)
 
        self._cached_shard_id = None
        self._cached_features = None
 
    def __len__(self):
        return len(self.shard_ids)
 
    def _load_shard(self, shard_id):
        if shard_id != self._cached_shard_id:
            path = os.path.join(self.shard_dir, f"{self.split}_shard_{shard_id:04d}.pt")
            self._cached_features = torch.load(path, weights_only=True)["features"]
            self._cached_shard_id = shard_id

        return self._cached_features
 
    def __getitem__(self, idx):
        shard_id = self.shard_ids[idx].item()
        local_id = self.local_ids[idx].item()
 
        shard = self._load_shard(shard_id)
        feature   = shard[local_id]
 
        return feature, (self.persons[idx], self.ages[idx])