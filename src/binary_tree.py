from abc import ABC, abstractmethod
from typing import Callable, List, Type

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from src.clustering.kmeans import KMeansImageClusterer
from src.router import map_emb_to_path


class PretrainedBinaryTreeRouter(ABC):
    @abstractmethod
    def get_path(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (data_dim,)
        Returns:
            path_mask: binary mask of shape (binary_tree_depth,)
        """


class RandomBinaryTreeRouter(PretrainedBinaryTreeRouter):
    def __init__(self, depth: int):
        self.depth = depth

    def get_path(self, x: torch.Tensor) -> torch.Tensor:
        return torch.randint(0, 2, (self.depth,))


class LatentVariableRouter(PretrainedBinaryTreeRouter):
    """
    Router which uses hierarchical clustering of learned (discrete) latent variable (using e.g., a VQ-VAE) to route data
    """
    def __init__(self, depth: int):
        self.depth = depth
        self.clusterer = None
        self.emb_to_path = []
    
    def compute_codebook(self, dataset: Dataset):
        """
        Args:
            dataset: expects (image, ...) tuples
        """
        self.clusterer = KMeansImageClusterer(n_clusters=2**(self.depth-1))
        self.clusterer.fit(dataset)
        
        codebook = self.clusterer.cluster_centers() # shape (n_clusters, n_features)
        codebook = torch.tensor(self.codebook)
        
        # Compute binary path for each codebook vector
        self.emb_to_path = map_emb_to_path(codebook, self.depth)
        
    def get_path(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (data_dim,)
        """
        
        # add a batch dimension
        x = x.unsqueeze(0)
        labels = self.clusterer.predict(x)
        emb_ix = labels[0].item()
        
        return self.emb_to_path[emb_ix]

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
        path_mask = path_mask[:, self._depth]
        left_idxs = torch.where(path_mask == 0)
        left_inputs = layer_output[left_idxs]
        right_idxs = torch.where(path_mask == 1)
        right_inputs = layer_output[right_idxs]

        final_output = torch.empty_like(x)

        if len(left_idxs) > 0:
            left_output = self.left_child(left_inputs, path_mask)
            final_output[left_idxs] = left_output
        if len(right_idxs) > 0:
            right_output = self.right_child(right_inputs, path_mask)
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
        self.router = router

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            y: (batch_size, output_dim)
        """
        path_mask = torch.vmap(self.router.get_path)(x)
        return self._root(x, path_mask)
