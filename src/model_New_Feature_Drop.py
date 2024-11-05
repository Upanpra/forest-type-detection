from torch import nn
import torch.nn.functional as F
from torch.nn.functional import dropout
from torch.nn import BatchNorm2d
from typing import Tuple


class DropoutCNN(nn.Module):
    """
    A small CNN that has dropout of input and hidden features
    """
    def __init__(self, input_size: Tuple[int, int, int] = (3, 128, 128), num_classes: int = 2):
        super(DropoutCNN, self).__init__()
        self.num_classes = num_classes

        self.downsample_factor: int = 8  # Factor of 8 bc we use 3 layers of max pooling. I.e. 2**3 = 8
        self.input_size = input_size
        assert self.input_size[1] % self.downsample_factor == 0
        assert self.input_size[2] % self.downsample_factor == 0

        # cnn(in_channel, out_channel, kernel_size, stride, padding)
        self.conv1 = nn.Conv2d(input_size[0], 32, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(64)
        #self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        #self.bn3 = nn.BatchNorm2d(128)
        # self.fc1 = nn.Linear(
        #     64 * int(self.input_size[1] / self.downsample_factor) * int(self.input_size[2] / self.downsample_factor),
        #     120)
        self.fc1 = nn.Linear(64, 64)
        # self.fc2 = nn.Linear(120, 84)
        self.dropout = nn.Dropout(0.2)  # note you had this at 0.5 before
        self.fc3 = nn.Linear(64, self.num_classes)

    def forward(self, x):
        # x = self.conv1(x)
        x = F.max_pool2d(F.relu(self.dropout(self.conv1(self.dropout(x)))), kernel_size=2, stride=2)
        x = F.max_pool2d(F.relu(self.dropout(self.conv2(x))), kernel_size=2, stride=2)
        #x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), kernel_size=2, stride=2)

        # x should be B, C, H, W:
        x = x.mean((-1, -2))
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
