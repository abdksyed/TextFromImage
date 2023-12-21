from typing import NamedTuple
from pathlib import Path

import numpy as np

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import albumentations as A

import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import LabelEncoder


class Classification(Dataset):
    def __init__(self, image_paths: str, resize: NamedTuple = None):
        self.image_paths = list(Path(image_paths).glob("*.png"))

        # Encoding the Target labels eg: b5f4 -> [1, 11, 15, 14]
        self.unique_characters = set()
        for image_path in self.image_paths:
            self.unique_characters.update([c for c in image_path.stem])
        self.unique_characters = {char:idx for idx, char in enumerate(self.unique_characters, start=1)}
        self.unique_characters[" "] = 0

        self.aug = A.Compose(
            [
                A.Resize(height=resize.IMAGE_HEIGHT, width=resize.IMAGE_WIDTH),
                A.Normalize(
                    mean=[0.7151951610105579]*3,  # Calculated from the Dataset
                    std=[0.009958559157242485]*3,  # Calculated from the Dataset
                    max_pixel_value=255.0,
                ),
            ]
        )

    def encode(self, labael):
        return [self.unique_characters[c] for c in labael]
    
    def decode(self, encoded):
        char2idx = {idx:char for char, idx in self.unique_characters.items()}
        prev = 0
        label = ""
        for idx in encoded:
            if idx != 0 and idx != prev:
                label += char2idx[idx]
            prev = idx
        return label
    
    def num_chars(self):
        return len(self.unique_characters)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = np.array(Image.open(image_path).convert("RGB"))
        image = self.aug(image=image)["image"]
        # Transpose the image to fit PyTorch's format (C, H, W)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "image": torch.tensor(image, dtype=torch.float), #.unsqueeze(0), # If Single Channel use unsqueeze
            "target": torch.tensor(self.encode(image_path.stem), dtype=torch.long
            ),
            "label": image_path.stem,
        }
