import os
import cv2
import torch
from torch.utils.data import Dataset

class RyersonEmotionDataset(Dataset):
    def __init__(self, dataset_dir):

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return (self.images[idx], self.labels[idx])
