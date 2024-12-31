import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ForestCNN(nn.Module):
    def __init__(self, input_size: Tuple[int, int, int, int] = (3, 64, 64, 64), num_classes: int = 2):
        super(ForestCNN, self).__init__()
        self.num_classes = num_classes

        #  self.downsample_factor: int = 4  # Adjust as needed
        self.input_size = input_size
        # assert self.input_size[0] % self.downsample_factor == 0
        # assert self.input_size[1] % self.downsample_factor == 0
        # assert self.input_size[2] % self.downsample_factor == 0

        # 3D convolution layers
        #  self.conv1 = nn.Conv3d(input_size[0], 32, kernel_size=3, padding=1)
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        # Add more convolutional layers as needed
        # self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        # self.bn3 = nn.BatchNorm3d(128)

        # Fully connected layers
        self.fc1 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, self.num_classes)

        self._print_input_shape_once = True

    def forward(self, x):
        if self._print_input_shape_once:
            print(f"ForestCNN input shape {x.shape}")
        if len(x.shape) == 4:
            # B, C, H, W
            x = x.unsqueeze(1)  # -> B, C, D, H, W
            # ( channels becomes depth so that we conv across channels)
        if self._print_input_shape_once:
            print(f"ForestCNN input shape after unsqueeze {x.shape}")
            self._print_input_shape_once = False
        x = F.max_pool3d(F.relu(self.bn1(self.conv1(x))), kernel_size=2, stride=2)
        x = F.max_pool3d(F.relu(self.bn2(self.conv2(x))), kernel_size=2, stride=2)
        # Add more max pooling and convolutional layers as needed
        # x = F.max_pool3d(F.relu(self.bn3(self.conv3(x))), kernel_size=2, stride=2)

        # Global average pooling across the spatial dimensions
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = x.view(x.size(0), -1)  # Flatten

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
