import torch
from torch.utils.data import Dataset

class ClipDataset(Dataset):
    def __init__(self, files_list, transform):
        self.files = files_list
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, key):
        sample = torch.load(self.files[key])

        frames = self.transform(sample["frames"])
        person = torch.tensor(sample["person"], dtype=torch.float32)
        age = torch.tensor(sample["age"], dtype=torch.float32)

        return frames, (person, age)