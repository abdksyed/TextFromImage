from collections import namedtuple
import logging

from dataset import Classification
from config import Config
from model import ConvRNN
from trainer import train_one_epoch, val_one_epoch, collate_fn
from utils import decode_preds

import torch

from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

g = torch.Generator().manual_seed(1337)

# Create a logger
logging.basicConfig(filename='./src/captcha/training.log', level=logging.INFO)

conf = Config(DATA_DIR="./data/captcha_images", DEVICE="cpu", NUM_WORKERS=0, EPOCHS=300)

print(f"Device used: {conf.DEVICE}")
logging.info(f"Device used: {conf.DEVICE}")

dataset = Classification(
    conf.DATA_DIR,
    resize=namedtuple("Resize", ["IMAGE_HEIGHT", "IMAGE_WIDTH"])(
        conf.IMAGE_HEIGHT, conf.IMAGE_WIDTH
    ),
)

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(
    dataset, [train_size, test_size], generator=g
)

logging.info(f"Train Dataset Length: {len(train_dataset)}")
logging.info(f"Test Dataset Length: {len(test_dataset)}")
# Log an example shape
logging.info(f"Example Image Shape: {train_dataset[0]['image'].shape}")
logging.info(f"Example Target Shape: {train_dataset[0]['target'].shape}")
logging.info(f"Example Target: {train_dataset[0]['target']}")
logging.info(f"Example Label: {train_dataset[0]['label']}")

# Creating Train and Test Loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=conf.BATCH_SIZE,
    num_workers=conf.NUM_WORKERS,
    shuffle=True,
    collate_fn=collate_fn,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=conf.BATCH_SIZE,
    num_workers=conf.NUM_WORKERS,
    shuffle=False,
    collate_fn=collate_fn,
)

print(len(train_loader), len(test_loader))
logging.info(f"Train Loader Length: {len(train_loader)}")
logging.info(f"Test Loader Length: {len(test_loader)}")

print(next(iter(train_loader))["images"].shape)
logging.info(f"Example Batch Shape: {next(iter(train_loader))['images'].shape}")
logging.info(f"Example Batch Target Shape: {next(iter(train_loader))['targets'].shape}")
logging.info(f"Example Batch Label: {next(iter(train_loader))['labels']}")

model = ConvRNN(num_chars=dataset.num_chars())

logging.info(f"Model: {model}")


criterion = torch.nn.CTCLoss(blank=0)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)

for epoch in range(conf.EPOCHS):
    train_loss, train_outputs, target_labels = train_one_epoch(
        model, train_loader, criterion, optimizer, conf.DEVICE
    )
    pred_labels = decode_preds(train_outputs, dataset.decode)
    logging.info(f"Predicted Labels: {pred_labels[:5]}")
    logging.info(f"Target Labels: {target_labels[:5]}")
    correct = sum([i == j for i, j in zip(pred_labels, target_labels)])
    print(
        f"Epoch: {epoch+1}, Train Loss: {train_loss}, Accuracy: {correct/len(target_labels)}"
    )
    logging.info(
        f"Epoch: {epoch+1}, Train Loss: {train_loss}, Accuracy: {correct/len(target_labels)}"
    )

    test_loss, test_outputs, target_labels = val_one_epoch(
        model, test_loader, criterion, conf.DEVICE
    )
    scheduler.step(test_loss)

    pred_labels = decode_preds(test_outputs, dataset.decode)
    logging.info(f"Predicted Labels: {pred_labels[:5]}")
    logging.info(f"Target Labels: {target_labels[:5]}")
    correct = sum([i == j for i, j in zip(pred_labels, target_labels)])
    print(
        f"Epoch: {epoch+1}, Test Loss: {test_loss}, Accuracy: {correct/len(target_labels)}"
    )
    logging.info(
        f"Epoch: {epoch+1}, Test Loss: {test_loss}, Accuracy: {correct/len(target_labels)}"
    )
