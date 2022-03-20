#
# Created by V.Sotskov on 19 March 2022
#
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.datasets import VisionDataset
from torch.utils.data import SubsetRandomSampler, DataLoader


class EmotionsDataset(VisionDataset):
    """
    Args:
        root (string): Root directory of dataset
        fname (string): name of the dataset file
        transform (callable, optional): A function/transform that takes in an image
            and returns a transformed version.
    """

    def __init__(
            self,
            root: str,
            fname: str,
            transform: Optional[Callable] = None):

        super().__init__(root, transform=transform)

        self.train_idxs = None
        self.data: Any = []
        self.targets = []

        path = (Path(root) / fname).resolve()
        print("Loading dataset...", end="\t")
        data = pd.read_csv(path)
        self.targets = data['emotion'].values
        x = data[' pixels'].str.split(' ', expand=True)
        x = x.astype('uint8').values
        self.data = x.reshape(-1, 1, 48, 48)  # one channel, 48x48 image
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        print("Done")

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.reshape(48, 48))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def split(self, *, ratio=0.8, batch_size=1, seed=123456789):
        """
        The function to split a dataset into train/test subsets
        Args:
            ratio (float): the ratio of train portion to test portion
            batch_size (int): size of batch to sample on each iteration
            seed (int): seed for random number generator
        """
        # define len of data
        n = len(self.data)

        # calculate the indexes
        rng = np.random.default_rng(seed=seed)
        self.train_idxs = rng.choice(np.arange(n), size=int(ratio * n), replace=False)
        ix_test = np.delete(np.arange(n), self.train_idxs)

        g_cpu = torch.Generator()
        g_cpu.manual_seed(seed)
        trainsmplr = SubsetRandomSampler(self.train_idxs, g_cpu)
        testsmplr = SubsetRandomSampler(ix_test, g_cpu)

        return DataLoader(self, batch_size=batch_size, sampler=trainsmplr), \
            DataLoader(self, batch_size=1, sampler=testsmplr)


class SimpleNet(nn.Module):

    def __init__(self, layers, activation, pooling=None):
        super().__init__()
        layer_list = []
        for layer in layers:
            layer_list.append(layer.pop('ltype')(**layer))
        self.layers = nn.ModuleList(layer_list)
        self.activation = activation
        self.pool = pooling

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if isinstance(self.layers[i-1], nn.Conv2d) and isinstance(layer, nn.Linear):
                x = torch.flatten(x, 1)
            x = layer(x)
            if i != len(self.layers)-1:
                x = self.activation(x)
                if isinstance(layer, nn.Conv2d) and self.pool is not None:
                    x = self.pool(x)
        return x


class AttentionalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3),
            nn.ReLU(True),
            nn.Conv2d(10, 10, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(810, 50)
        self.fc2 = nn.Linear(50, 7)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 10 * 10, 48),
            nn.ReLU(True),
            nn.Linear(48, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 10 * 10)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = self.conv(x)
        x = self.conv2_drop(self.conv(x))
        x = x.view(-1, 810)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
