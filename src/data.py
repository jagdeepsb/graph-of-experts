import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from tqdm import tqdm


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


class FixedSizeWrapper(Dataset):
    def __init__(self, dataset: Dataset, size: int):
        self.dataset = dataset
        self.size = size
        
        if size > len(dataset):
            raise ValueError(f"FixedSizeWrapper size {size} is larger than the dataset size {len(dataset)}")
        self.samples = np.random.choice(len(self.dataset), size=self.size, replace=False)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.dataset[self.samples[idx]]
    
class CelebADataset(Dataset):
    def __init__(self, root: str = "./data", train: bool = True, download: bool = True, n=6000):
        """
        Create an extended CelebA dataset split along 3 attributes.

        Args:
            root (str): Root directory of dataset
            train (bool): If True, creates dataset from training set, else test set
            download (bool): If True, downloads the dataset if not already present
        """
        split = 'train' if train else 'test'

        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        dataset = datasets.CelebA(root=root, split=split, download=download, transform=transform)
        
        self.path_attributes = ['Black_Hair', 'Male', 'Eyeglasses']
        self.num_paths = int(len(self.path_attributes)**3)
        self.attr_dict = {}
        for i in range(2**len(self.path_attributes)):
            attr_str = ''
            for j, attr in enumerate(self.path_attributes):
                if i & (1 << j):
                    attr_str += f'{attr}; '
                else:
                    attr_str += f'not {attr}; '
            self.attr_dict[i] = attr_str.strip()
        
        self.class_attribute = 'Smiling'

        # data split
        n = n // 2
        d1 = [] 
        attr1 = np.zeros((n, len(self.path_attributes)))
        class1 = np.zeros((n,))
        d2 = [] 
        attr2 = np.zeros((n, len(self.path_attributes)))
        class2 = np.zeros((n,))
        split_attr_idx = dataset.attr_names.index(self.path_attributes[0])
        for x, y in tqdm(dataset, desc=f'Splitting data by {self.path_attributes[0]}'):
            if y[split_attr_idx] == 1 and len(d1) < n:
                d1.append(x)
                for i, attr in enumerate(self.path_attributes):
                    attr_idx = dataset.attr_names.index(attr)
                    attr1[len(d1)-1, i] = y[attr_idx]
                class1[len(d1)-1] = y[dataset.attr_names.index(self.class_attribute)]
            elif y[split_attr_idx] == 0 and len(d2) < n:
                d2.append(x)
                for i, attr in enumerate(self.path_attributes):
                    attr_idx = dataset.attr_names.index(attr)
                    attr2[len(d2)-1, i] = y[attr_idx]
                class2[len(d2)-1] = y[dataset.attr_names.index(self.class_attribute)]

            if len(d1) >= n and len(d2) >= n:
                break

        self.x = d1 + d2
        self.attr = np.vstack((attr1, attr2))
        self.target = np.hstack((class1, class2))

    def __len__(self):
        """
        Total number of samples
        """
        return len(self.x)

    def __getitem__(self, idx):
        """
        Retrieve an item by index.

        Returns:
            tuple: (image, attr (int), class (int))
        """
        image = self.x[idx]
        attr = 0
        for i, attr_val in enumerate(self.attr[idx]):
            attr += attr_val * 2**i
        target = self.target[idx]

        return (
            image,
            attr,
            target,
        )