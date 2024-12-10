from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Type

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.clustering.kmeans import KMeansImageClusterer, KMeansVQVAEClusterer
from src.router import get_binary_path, map_emb_to_path


class PretrainedBinaryTreeRouter(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_path(self, x: torch.Tensor, **metadata_kwargs) -> torch.Tensor:
        """
        Args:
            x: (data_dim,)
        Returns:
            path_mask: binary mask of shape (binary_tree_depth,)
        """


class RandomBinaryTreeRouter(PretrainedBinaryTreeRouter):
    def __init__(self, depth: int):
        super().__init__()
        self.depth = depth
        self.sample_to_path = {}

    def compute_codebook(self, dataset: Dataset):
        """
        Randomly assigns each datapoint in the dataset to a binary path.
        Args:
            dataset: expects (image, label, ...) tuples or something similar
        """
        n_paths = 2 ** (self.depth - 1)

        for el in dataset:
            assert type(el) == tuple
            image = el[0]
            assert (
                type(image) == torch.Tensor
            ), f"Expected torch.Tensor, got {type(image)}"
            image = image.numpy()

            path_idx = torch.randint(0, n_paths, (1,)).item()

            path = get_binary_path(self.depth, path_idx)

            key = image.tobytes()
            self.sample_to_path[key] = path

    def _get_unbached_path(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get path for a single unbatched sample
        """
        key = x.cpu().numpy().tobytes()

        if key not in self.sample_to_path:  # inference time
            # gnereate a random path
            path_idx = torch.randint(0, 2 ** (self.depth - 1), (1,)).item()
            path = get_binary_path(self.depth, path_idx)
            # print(f"Generated random path: {path}")
            return path

        # train time, return the pre-assigned path
        # print(f"Retrieved pre-assigned path: {self.sample_to_path[key]}")
        return self.sample_to_path[key]

    def get_path(self, x: torch.Tensor, **metadata_kwargs) -> torch.Tensor:
        """
        Retrieve the pre-assigned path for the given sample tensor x.
        """
        paths = [self._get_unbached_path(x[i]) for i in range(x.shape[0])]
        return torch.stack(paths)


class MNISTOracleRouter(PretrainedBinaryTreeRouter):
    """
    Oracle router which predicts a path based on the rotation label
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("mask_arr", torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]]))

    def get_path(
        self, x: torch.Tensor, oracle_metadata: torch.Tensor, **metadata_kwargs
    ):
        """
        Args:
        - x: (bs, ...)
        - rotation_labels: (bs,...)
        """
        return self.mask_arr[oracle_metadata]

class CelebAOracleRouter(PretrainedBinaryTreeRouter):
    """
    Oracle router which predicts a path based on the rotation label
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("mask_arr", torch.Tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]))

    def get_path(
        self, x: torch.Tensor, comb_attr_labels: torch.Tensor, **metadata_kwargs
    ):
        """
        Args:
        - x: (bs, ...)
        - comb_attr_labels: (bs,...)
        """
        return self.mask_arr[comb_attr_labels]

class LatentVariableRouter(PretrainedBinaryTreeRouter):
    """
    Router which uses hierarchical clustering of learned (discrete) latent variable (using e.g., a VQ-VAE) to route data
    """

    def __init__(self, depth: int):
        super().__init__()
        self.depth = depth
        self.clusterer = None
        self.emb_to_path = []

    def compute_codebook(self, dataset: Dataset):
        """
        Args:
            dataset: expects (image, ...) tuples
        """
        # self.clusterer = KMeansImageClusterer(n_clusters=2 ** (self.depth - 1))
        self.clusterer = KMeansVQVAEClusterer(n_clusters=2 ** (self.depth - 1))
        self.clusterer.fit(dataset)

        codebook = self.clusterer.cluster_centers()  # shape (n_clusters, n_features)
        codebook = torch.tensor(codebook)

        # Compute binary path for each codebook vector
        self.emb_to_path = map_emb_to_path(codebook, self.depth)

    def get_path(self, x: torch.Tensor, **metadata_kwargs) -> torch.Tensor:
        """
        x: (bs, data_dim)
        """
        labels = self.clusterer.predict(x.cpu())
        paths = [self.emb_to_path[labels[i].item()] for i in range(x.shape[0])]
        return torch.stack(paths)


class BinaryTreeNode:
    def __init__(
        self,
        modules_by_depth: List[Type[nn.Module]],
        depth: int,
        name: str,
        register_hook: Callable[[str, nn.Module], None],
    ):
        assert len(modules_by_depth) > 0
        self._depth = depth
        self._module = modules_by_depth[0]()
        self._has_children = len(modules_by_depth) > 1
        register_hook(f"Node{name}", self._module)
        if self._has_children:
            self.left_child = BinaryTreeNode(
                modules_by_depth[1:], depth + 1, name + "0", register_hook
            )
            self.right_child = BinaryTreeNode(
                modules_by_depth[1:], depth + 1, name + "1", register_hook
            )

    def __call__(self, x: torch.Tensor, path_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, data_dim)
            path_mask: binary mask of shape (batch_size, binary_tree_depth)
        Returns:
            y: (batch_size, data_dim)
        """
        # Process current layer
        layer_output = self._module(x)

        if not self._has_children:
            return layer_output

        # Route to children if necessary
        layer_mask = path_mask[:, self._depth]
        left_idxs = torch.where(layer_mask == 0)
        right_idxs = torch.where(layer_mask == 1)

        left_inputs = layer_output[left_idxs]
        right_inputs = layer_output[right_idxs]

        left_output = None
        right_output = None

        if len(left_idxs) > 0:
            left_output = self.left_child(left_inputs, path_mask[left_idxs])
        if len(right_idxs) > 0:
            right_output = self.right_child(right_inputs, path_mask[right_idxs])

        output = left_output if left_output is not None else right_output
        output_shape = (x.shape[0],) + output.shape[1:]
        final_output = torch.empty(output_shape).to(x)
        if left_output is not None:
            final_output[left_idxs] = left_output
        if right_output is not None:
            final_output[right_idxs] = right_output

        return final_output


class BinaryTreeGoE(nn.Module):
    """
    Graph of Experts with binary tree graph structure
    """

    def __init__(
        self,
        modules_by_depth: List[Type[nn.Module]],
        router: PretrainedBinaryTreeRouter,
    ):
        super().__init__()
        self._depth = len(modules_by_depth)
        self._root = BinaryTreeNode(modules_by_depth, 0, "", self.register_module)
        self.register_module("router", router)

    def forward(
        self, x: torch.Tensor, router_metadata: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            y: (batch_size, output_dim)
        """
        path_mask = self.router.get_path(x, **router_metadata)
        return self._root(x, path_mask)
