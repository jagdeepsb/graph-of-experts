import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


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
