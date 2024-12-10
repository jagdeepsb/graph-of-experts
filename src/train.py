import os
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.data import FixedSizeWrapper, RotatedMNISTDataset


def do_train_epoch(model, loader, optimizer, epoch, device) -> float:
    model.train()
    epoch_losses = []
    for idx, (x, oracle_metadata, y) in enumerate(loader):
        x = x.to(device)
        oracle_metadata = oracle_metadata.to(device)
        y = y.to(device)

        metadata = {
            "oracle_metadata": oracle_metadata,
        }

        optimizer.zero_grad()
        logits = model(x, router_metadata=metadata)  # (bs, num_digit_classes)
        loss = cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    epoch_loss = torch.mean(torch.Tensor(epoch_losses)).item()
    # print(f'Train epoch {epoch} loss: {epoch_loss:.4f}')
    return epoch_loss


def do_val_epoch(model, loader, epoch, device) -> float:
    model.eval()
    epoch_losses = []
    for idx, (x, oracle_metadata, y) in tqdm(enumerate(loader)):
        x = x.to(device)
        oracle_metadata = oracle_metadata.to(device)
        y = y.to(device)

        metadata = {
            "oracle_metadata": oracle_metadata,
        }

        logits = model(x, router_metadata=metadata)  # (bs, num_digit_classes)
        loss = cross_entropy(logits, y)
        epoch_losses.append(loss.item())
    epoch_loss = torch.mean(torch.Tensor(epoch_losses)).item()
    # print(f'Val epoch {epoch} loss: {epoch_loss:.4f}')
    return epoch_loss


def get_accuracy(model, loader, device):
    model.eval()
    total_samples = 0
    total_correct = 0
    for idx, (x, oracle_metadata, y) in enumerate(loader):
        x = x.to(device)
        oracle_metadata = oracle_metadata.to(device)
        y = y.to(device)  # (bs,)

        metadata = {
            "oracle_metadata": oracle_metadata,
        }

        logits = model(x, router_metadata=metadata)  # (bs, num_digit_classes)
        prediction = torch.argmax(logits, dim=1)  # (bs,)
        total_samples += prediction.shape[0]
        total_correct += torch.sum(prediction == y).item()
    accuracy = total_correct / total_samples * 100
    return accuracy


def training_loop(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 100,
    lr: float = 0.005,
    epochs_per_accuracy: int = 5,
) -> List[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    val_losses = []
    accuracy = None
    # make a pbar displaying the epoch number, training loss, and validation loss, and test accuracy
    pbar = tqdm(
        range(num_epochs), desc="Epochs", total=num_epochs, leave=False, position=0
    )

    for epoch in pbar:
        train_loss = do_train_epoch(
            model, train_loader, optimizer, epoch, device=device
        )
        val_loss = do_val_epoch(model, val_loader, epoch, device=device)
        val_losses.append(val_loss)
        if epoch % epochs_per_accuracy == 0:
            accuracy = get_accuracy(model, test_loader, device=device)

        pbar.set_postfix(
            {"Train loss": train_loss, "Val loss": val_loss, "Test acc": accuracy}
        )
    return val_losses


def save_model(model: nn.Module, save_path: str):
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)


def load_model(model: nn.Module, save_path: str):
    return model.load_state_dict(torch.load(save_path))
