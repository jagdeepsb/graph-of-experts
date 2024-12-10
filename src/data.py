from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


class FixedSizeWrapper(Dataset):
    def __init__(self, dataset: Dataset, size: int):
        self.dataset = dataset
        self.size = size

        if size > len(dataset):
            raise ValueError(
                f"FixedSizeWrapper size {size} is larger than the dataset size {len(dataset)}"
            )
        self.samples = np.random.choice(
            len(self.dataset), size=self.size, replace=False
        )

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.dataset[self.samples[idx]]


class RotatedMNISTDataset(Dataset):
    def __init__(self, root: str = "./data", train: bool = True, download: bool = True):
        """
        Create an extended MNIST dataset with rotated variations.

        Args:
            root (str): Root directory of dataset
            train (bool): If True, creates dataset from training set, else test set
            download (bool): If True, downloads the dataset if not already present
            transform (callable, optional): Optional transform to be applied on an image
        """
        self.mnist = datasets.MNIST(root=root, train=train, download=download)
        self.rotations = [0, 90, 180, 270]

    def __len__(self):
        """
        Total number of samples: original MNIST images * number of rotations
        """
        return len(self.mnist) * len(self.rotations)

    def __getitem__(self, idx):
        """
        Retrieve an item by index.

        Returns:
            tuple: (rotated_image, rotation_class, digit_class)
        """
        # Calculate original MNIST index and rotation index
        original_idx = idx // len(self.rotations)
        rotation_idx = idx % len(self.rotations)

        # Get original image and label
        image, digit_class = self.mnist[original_idx]

        # Apply rotation
        rotation_angle = self.rotations[rotation_idx]
        rotated_image = transforms.functional.rotate(image, rotation_angle)

        # cast image to tensor
        rotated_image = transforms.functional.to_tensor(rotated_image)

        if rotation_idx == 0:
            # invert the colors in the image
            max_value = rotated_image.max()
            rotated_image = max_value - rotated_image

        # Return image, rotation class, and original digit class
        return (
            rotated_image,
            rotation_idx,  # 0 for 0°, 1 for 90°, 2 for 180°, 3 for 270°
            digit_class,
        )

    @classmethod
    def get_rotated_mnist_loaders(
        cls,
        downsample_factor: int = 20,
    ) -> Tuple[Dataset, Dataset, Dataset, DataLoader, DataLoader, DataLoader]:
        # Initialize dataset
        dataset = cls()

        # Assuming `dataset` is your PyTorch Dataset
        dataset_size = len(dataset)
        train_size = int(0.7 * dataset_size)
        val_size = int(0.2 * dataset_size)
        test_size = dataset_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(40),
        )

        train_dataset = FixedSizeWrapper(
            dataset=train_dataset, size=train_size // downsample_factor
        )
        val_dataset = FixedSizeWrapper(
            dataset=val_dataset, size=val_size // downsample_factor
        )
        test_dataset = FixedSizeWrapper(
            dataset=test_dataset, size=test_size // downsample_factor
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        return (
            train_dataset,
            val_dataset,
            test_dataset,
            train_loader,
            val_loader,
            test_loader,
        )

