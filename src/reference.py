from typing import List, Type, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReferenceModel(nn.Module):
    """
    One large model. The architecture mimics a forward pass along a given path through the GoE DAG.
    """

    def __init__(self, modules_by_depth: List[Type[nn.Module]]):
        super().__init__()
        self.net = nn.Sequential(*[module() for module in modules_by_depth])

    def forward(self, x: torch.Tensor, router_metadata: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
        - x: (batch_size, 1, 28, 28)
        """
        # print(type(x))
        # print(metadata_kwargs)
        return self.net(x)
