from typing import List

import torch.nn as nn
from torch import Tensor


class MLP(nn.Module):
    def __init__(self, unit_dims: List[int], activation: nn.Module = nn.SiLU):
        super().__init__()
        self.unit_dims = unit_dims
        self.activation = activation
        self.mlp = self.build()

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)

    def build(self) -> nn.Module:
        mlp = []
        for idx in range(len(self.unit_dims) - 1):
            mlp.append(nn.Linear(self.unit_dims[idx], self.unit_dims[idx + 1]))
            mlp.append(self.activation())
        return nn.Sequential(*mlp)
