import os
import pathlib
import typing

import torch
import torchvision


class CycleGANDataset(torchvision.datasets.ImageFolder):
    """
    Dataset for CycleGAN model.
    """

    def __init__(
        self,
        images_root: str | pathlib.Path,
        transform: typing.Callable = None,
        seed: int = None,
    ) -> None:
        """
        :param images_root: root directory for images. Should contain only two subdirectories
        :param transform: method for transforming images
        :param seed: random generator seed used in images pairs making
        """
        assert (
            len(os.listdir(images_root)) == 2
        ), "Images root should have exactly 2 subdirectories, that will be used as CycleGAN domains"

        super().__init__(images_root, transform)
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

        self.pairs = None
        self.reset_pairs()

    def reset_pairs(self) -> None:
        """
        Method for resetting datasets pairs to make them different in each epoch
        """
        self.pairs = torch.randperm(len(self.first_class), generator=self.generator)

    def __len__(self) -> int:
        """
        Returns length of a dataset. Always returns length of the smallest class
        :return: Length of the smallest class
        """
        return len(self.first_class)

    def __getitem__(self, index: int) -> tuple[typing.Any, typing.Any]:
        """
        Method reads images located and `index` position and its pair. Transforms them if necessary.
        :param index: sample index
        :return: the pair of images
        """
        first_path = self.first_class[index]
        second_path = self.second_class[self.pairs[index]]

        first_sample = self.loader(first_path)
        second_sample = self.loader(second_path)

        if self.transform is not None:
            first_sample = self.transform(first_sample)
            second_sample = self.transform(second_sample)

        return first_sample, second_sample
