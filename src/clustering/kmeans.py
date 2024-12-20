import os

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

from src.vqvae.train import get_vae, train_vae


class KMeansClusterer(nn.Module):
    def __init__(self, n_clusters, verbose=False, n_tries=10):
        super().__init__()
        self.n_clusters = n_clusters
        self.model = KMeans(
            n_clusters=n_clusters,
            verbose=1 if verbose else 0,
            n_init=n_tries,
        )
        self._cluster_centers = None

    def fit(self, X: np.ndarray) -> None:
        """
        X: numpy array of shape (n_samples, n_features)
        """
        self.model.fit(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        X: numpy array of shape (n_samples, n_features)
        returns:
        - labels: cluster id of each sample; numpy array of shape (n_samples,)
        """
        ids = self.model.predict(X)
        # cluster_centers = self._cluster_centers[ids]
        return ids

    def cluster_centers(self) -> np.ndarray:
        """
        returns: numpy array of shape (n_clusters, n_features)
        """
        return self.model.cluster_centers_


class KMeansImageClusterer(KMeansClusterer):
    def fit(self, dataset: Dataset):
        """
        dataset: torch.utils.data.Dataset
        Expects (image, ...) tuples
        """

        features = []
        print(f"Computing kmeans features for {len(dataset)} images")
        for el in tqdm(dataset):
            assert type(el) == tuple
            image = el[0]
            assert (
                type(image) == torch.Tensor
            ), f"Expected torch.Tensor, got {type(image)}"
            image = image.numpy()
            image = self.preprocess_image(image)

            features.append(image.flatten())

        features = np.array(features)
        print(f"Fitting kmeans model with {features.shape[0]} samples")
        self.model.fit(features)
        self._cluster_centers = self.model.cluster_centers_

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        x: torch.Tensor of shape (n_samples, C, H, W)
        returns: numpy array of shape (n_samples,)
        """
        features = []
        for i in range(x.shape[0]):
            image = x[i].cpu().numpy()
            image = self.preprocess_image(image)
            features.append(image.flatten())

        features = np.array(features)
        ids = self.model.predict(features)
        return ids

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        image: numpy array of shape (C, H, W)
        returns: numpy array of shape (n_features,)
        """

        # if the image has an alpha channel, remove it
        if image.shape[0] == 4:
            image = image[:3]

        # if the image is grayscale, remove the channel dimension
        if image.shape[0] == 1:
            image = image[0]

        resized_image = cv2.resize(image, (4, 4), interpolation=cv2.INTER_CUBIC)
        return resized_image

    def debug(self, dataset: Dataset):
        """

        dataset: torch.utils.data.Dataset
        Expects (image, ...) tuples
        """

        features, raw_images = [], []
        for el in dataset:
            assert type(el) == tuple
            image = el[0]
            assert type(image) == torch.Tensor
            image = image.numpy()
            raw_images.append(image.copy())

            image = self.preprocess_image(image)
            features.append(image.flatten())

        features = np.array(features)
        labels = self.model.predict(features)

        n_per = 20
        label_to_images = {i: [] for i in range(self.n_clusters)}
        for image, label in zip(raw_images, labels):
            if len(label_to_images[label]) < n_per:
                label_to_images[label].append(image)

        # Plot images
        fig, axes = plt.subplots(self.n_clusters, n_per, figsize=(80, 80))
        for i in range(self.n_clusters):
            for j in range(n_per):
                img = label_to_images[i][j]
                img = np.transpose(img, (1, 2, 0))
                img = (img + 1) / 2
                axes[i, j].imshow(img)
                axes[i, j].axis("off")
        plt.tight_layout()

        # Save the plot
        plt.savefig("clustered_images.png")


class KMeansVQVAEClusterer(KMeansClusterer):
    def __init__(
        self,
        n_clusters,
        dataset: Dataset,
        experiment_type: str,
        verbose=False,
        n_tries=10,
    ):
        super().__init__(n_clusters, verbose, n_tries)
        self.vae = None

        # determine number of channels in input dim
        input_dim = dataset[0][0].shape[0]
        self.vae = get_vae(input_dim=input_dim, experiment_type=experiment_type)

    def fit(self, dataset: Dataset):
        """
        dataset: torch.utils.data.Dataset
        Expects (image, ...) tuples
        """

        # train the vae
        if os.path.exists(self.vae.get_save_path()):
            print("Loading pre-trained VAE model...")
            self.vae.load_state_dict(torch.load(self.vae.get_save_path()))
        else:
            print("Training VAE model...")
            train_vae(self.vae, dataset)

        # get the embeddings
        device = next(self.vae.parameters()).device
        features = []
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
        for el in tqdm(dataloader):
            x = el[0]
            x = x.to(device)
            feat = self.vae.get_embedding(x)

            # flatten embeddings except for the batch dimension
            feat = feat.view(feat.size(0), -1)
            features.append(feat.cpu().detach().numpy())

        features = np.concatenate(features, axis=0)

        print(
            f"Fitting kmeans model with {features.shape[0]} samples matrix size {features.shape}"
        )
        self.model.fit(features)
        self._cluster_centers = self.model.cluster_centers_

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        x: torch.Tensor of shape (n_samples, C, H, W)
        returns: numpy array of shape (n_samples,)
        """

        features = self.vae.get_embedding(x)
        features = features.view(features.size(0), -1)
        features = features.cpu().detach().numpy()
        ids = self.model.predict(features)
        return ids

    def debug(self, dataset: Dataset):
        """

        dataset: torch.utils.data.Dataset
        Expects (image, ...) tuples
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        features, raw_images = [], []
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
        for el in tqdm(dataloader):
            assert type(el) == tuple or type(el) == list
            image = el[0]
            assert type(image) == torch.Tensor
            image = image.numpy()
            raw_images.extend([im.copy() for im in image])

            x = torch.tensor(image).to(device)
            feat = self.vae.get_embedding(x)
            feat = feat.view(feat.size(0), -1)
            features.append(feat.cpu().detach().numpy())

        features = np.concatenate(features, axis=0)
        labels = self.model.predict(features)

        n_per = 20
        label_to_images = {i: [] for i in range(self.n_clusters)}
        for image, label in zip(raw_images, labels):
            if len(label_to_images[label]) < n_per:
                label_to_images[label].append(image)

        # Plot images
        fig, axes = plt.subplots(self.n_clusters, n_per, figsize=(80, 20))
        for i in range(self.n_clusters):
            for j in range(n_per):
                img = label_to_images[i][j]
                img = np.transpose(img, (1, 2, 0))
                img = (img + 1) / 2
                axes[i, j].imshow(img)
                axes[i, j].axis("off")
        plt.tight_layout()

        # Save the plot
        plt.savefig("clustered_images.png")
