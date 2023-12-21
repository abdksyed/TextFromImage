from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    images = [i["image"] for i in batch]
    targets = [i["target"] for i in batch]
    labels = [i["label"] for i in batch]

    images = torch.stack(images)
    # Pad the targets to the same length
    targets = pad_sequence(targets, batch_first=True)

    return {"images": images, "targets": targets, "labels": labels}


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: str | torch.device,
):
    model = model.to(device)
    model.train()

    train_loss = 0.0
    outputs = []
    all_labels = []
    for batch in tqdm(train_loader):
        images, targets, labels = batch["images"], batch["targets"], batch["labels"]
        all_labels.extend(labels)
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()

        output = model(images)  # (bs, seq_len, num_classes)
        outputs.append(output)
        # CTC Loss Expects the shape of the predictions to be (T, N, C)
        # T -> maximum length of the sequences in your batch
        # N -> the batch size
        # C -> the number of classes (including the blank label)
        # bs, seq_len, num_classes) -> (seq_len, bs, num_classes) or (T, N, C)
        output = output.permute(1, 0, 2)

        # For CTC loss, we require input_lengths(model output) and target_lengths(actual target)
        # By input_length we mean the length of the sequence of the output of the model
        # By target_length we mean the length of the sequence of the actual target
        input_lengths = torch.full(
            size=(output.shape[1],),  # (N,) or (batch_size)
            fill_value=output.shape[
                0
            ],  # Each value is seq_len, fixed in our case `T` or `seq_len`
            dtype=torch.long,
            device=device,
        )
        target_lengths = torch.count_nonzero(targets, dim=-1)

        loss = criterion(output, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        # accumulate training loss
        train_loss += loss.item()

    stacked_outputs = torch.stack(outputs, dim=0) # (num_iters, bs, seq_len, num_classes)
    # stacked_outputs -> (bs, seq_len, num_classes)
    stacked_outputs = stacked_outputs.view(-1, stacked_outputs.shape[-2], stacked_outputs.shape[-1])

    return train_loss / len(train_loader), stacked_outputs, all_labels


def val_one_epoch(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: str | torch.device,
):
    model = model.to(device)
    model.eval()

    test_loss = 0.0
    outputs = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images, targets, labels = batch["images"], batch["targets"], batch["labels"]
            all_labels.extend(labels)
            images, targets = images.to(device), targets.to(device)

            output = model(images)  # (bs, seq_len, num_classes)
            outputs.append(output)
            # CTC Loss Expects the shape of the predictions to be (T, N, C)
            # T -> maximum length of the sequences in your batch
            # N -> the batch size
            # C -> the number of classes (including the blank label)
            # bs, seq_len, num_classes) -> (seq_len, bs, num_classes) or (T, N, C)
            output = output.permute(1, 0, 2)

            # For CTC loss, we require input_lengths(model output) and target_lengths(actual target)
            # By input_length we mean the length of the sequence of the output of the model
            # By target_length we mean the length of the sequence of the actual target
            input_lengths = torch.full(
                size=(output.shape[1],),  # (N,) or (batch_size)
                fill_value=output.shape[
                    0
                ],  # Each value is seq_len, fixed in our case `T` or `seq_len`
                dtype=torch.long,
                device=device,
            )
            target_lengths = torch.full(
                size=(targets.shape[0],),  # (N,) or (batch_size)
                fill_value=targets.shape[
                    1
                ],  # Each value is the max length of the target in the batch.
                dtype=torch.long,
                device=device,
            )

            loss = criterion(output, targets, input_lengths, target_lengths)

            # accumulate test loss
            test_loss += loss.item()

        stacked_outputs = torch.stack(outputs, dim=0) # (num_iters, bs, seq_len, num_classes)
        # stacked_outputs -> (bs, seq_len, num_classes)
        stacked_outputs = stacked_outputs.view(-1, stacked_outputs.shape[-2], stacked_outputs.shape[-1])
    
    return test_loss / len(test_loader), stacked_outputs, all_labels
