import os
import pathlib
import typing

import torch
import torchvision


class CycleGANDataset(torchvision.datasets.ImageFolder):
    """Dataset for CycleGAN model."""

    def __init__(
        self, images_root: str | pathlib.Path, transform: typing.Callable = None, seed: int = None, **kwargs: typing.Any
    ) -> None:
        """
        Parameters
        ----------
        images_root : str | pathlib.Path
            Root directory for images. Should contain only two subdirectories
        transform : typing.Callable, default: None
            Method for transforming images
        seed : int, default: None
            Random generator seed used in images pairs making
        **kwargs : typing.Any
            Parameters for torchvision.datasets.ImageFolder constructor
        """
        assert (
            len(os.listdir(images_root)) == 2
        ), "Images root should have exactly 2 subdirectories, that will be used as CycleGAN domains"

        super().__init__(images_root, transform, **kwargs)
        self.first_class = []
        self.second_class = []
        for sample, label in self.samples:
            if label == 0:
                self.first_class.append(sample)
            else:
                self.second_class.append(sample)

        if len(self.second_class) < len(self.first_class):  # Make sure first_class is always be the smallest class
            self.first_class, self.second_class = self.second_class, self.first_class

        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)
        else:
            self.generator.seed()  # If this method is not called, then all instances will have the same value as seed

        self.pairs = None
        self.reset_pairs()

    def reset_pairs(self) -> None:
        """
        Method for resetting datasets pairs to make them different in each epoch

        Returns
        -------
        None
        """
        self.pairs = torch.randperm(len(self.first_class), generator=self.generator)

    def __len__(self) -> int:
        """
        Returns length of a dataset. Always returns length of the smallest class

        Returns
        -------
        int
            Length of the smallest class
        """
        return len(self.first_class)

    def __getitem__(self, index: int) -> tuple[typing.Any, typing.Any]:
        """
        Method reads images located and `index` position and its pair. Transforms them if necessary.

        Parameters
        ----------
        index : int
            Sample index

        Returns
        -------
        typing.Any
            First image from pair
        typing.Any
            Second image from pair
        """
        first_path = self.first_class[index]
        second_path = self.second_class[self.pairs[index]]

        first_sample = self.loader(first_path)
        second_sample = self.loader(second_path)

        if self.transform is not None:
            first_sample = self.transform(first_sample)
            second_sample = self.transform(second_sample)

        return first_sample, second_sample
