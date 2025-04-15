import os
import torch
import glob
from torch.utils.data import Dataset
from torchvision.io import read_image

class RyersonEmotionDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.length = 0
        self.dataset = {}
        self.label_map = {
            'ha': 0,
            'sa': 1,
            'an': 2,
            'fe': 3,
            'su': 4,
            'di': 5,
        }
        for _, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.jpg'):
                    self.length += 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx not in self.dataset:
            img_path = glob.glob(f'{idx}??.jpg', root_dir=self.root_dir)[0]
            label = self.label_map[img_path[-6:-4]]
            image = read_image(os.path.join(self.root_dir, img_path))
            image = image.float() / 255.0
            self.dataset[idx] = (image, label)
        return self.dataset[idx]

