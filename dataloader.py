import os
import cv2
import torch
from torch.utils.data import Dataset

class RyersonEmotionDataset(Dataset):
    def __init__(self, path):
        label_map = {
            'happy': 0,
            'sad': 1,
            'angry': 2,
            'fear': 3,
            'surpise': 4,
            'disgust': 5,
        }
        self.images = [];
        for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.avi'):
                vid_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        #some cv2 function to read the image
        #take last part to get label
        face_normalized = face_resized.astype('float32') / 255.0
        image_as_tensor = torch.unsqueeze(torch.tensor(face_normalized), 0) 
        return (self.images[idx], self.labels[idx])
