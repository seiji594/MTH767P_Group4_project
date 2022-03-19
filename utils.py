#
# Created by V.Sotskov on 19 March 2022
#
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Any, Tuple

import torch
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
        print("Done")
        self.targets = data['emotion'].values
        x = data[' pixels'].str.split(' ', expand=True)
        x = x.astype('uint8').values
        self.data = x.reshape(-1, 1, 48, 48)  # one channel, 48x48 image
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

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
        self.train_idxs = rng.integers(n, size=int(ratio * n))
        ix_test = np.delete(np.arange(n), self.train_idxs)

        trainsmplr = SubsetRandomSampler(self.train_idxs, torch.Generator())
        testsmplr = SubsetRandomSampler(ix_test, torch.Generator())

        return DataLoader(self, batch_size=batch_size, sampler=trainsmplr), \
            DataLoader(self, batch_size=batch_size, sampler=testsmplr)
