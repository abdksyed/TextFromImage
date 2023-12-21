from dataclasses import dataclass

import torch


@dataclass
class Config:
    DATA_DIR: str = "../../data/captcha_images"
    BATCH_SIZE: int = 8
    IMAGE_WIDTH: int = 300
    IMAGE_HEIGHT: int = 75
    NUM_WORKERS: int = 4
    EPOCHS: int = 100
    DEVICE: str | torch.device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
