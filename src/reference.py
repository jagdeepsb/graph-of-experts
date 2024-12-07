import torch
import torch.nn as nn
import torch.nn.functional as F


class RotatedMNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor):
        """
        Args:
        - x: (batch_size, 1, 28, 28)
        """
        # First module
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # (batch_size, 32, 14, 14)
        # Second module
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # (batch_size, 64, 7, 7)

        # Third module
        x = x.view(-1, 64 * 7 * 7)  # (batch_size, 64 * 7 * 7)
        x = F.relu(self.fc1(x))  # (batch_size, 128)
        logits = self.fc2(x)  # (batch_size, 10)
        return logits
