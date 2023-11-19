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

    def reset_pairs(self, training_indexes: typing.Sequence[int] = None) -> None:
        """
        Method for resetting datasets pairs to make them different in each training epoch

        Parameters
        ----------
        training_indexes : list[int], default: None
            When this value is passed, only indexes in this list will have assigned a new pair

        Returns
        -------
        None
        """
        if training_indexes is not None:
            assert self.pairs is not None, "`reset_pairs` with training indexes can be call only after first full reset"
            assert len(training_indexes) <= len(self), "Number of training indexes is larger than size of the dataset."
            assert min(training_indexes) >= 0 and max(training_indexes) < len(self), "Invalid training_indexes"
            blocked_pairs = [pair for index, pair in enumerate(self.pairs) if index not in training_indexes]
        else:
            training_indexes = range(len(self.first_class))  # Will reset pairs for the whole dataset
            blocked_pairs = None

        candidates = torch.ones(len(self.second_class))  # Initialize probability for all samples as 1
        if blocked_pairs is not None:
            candidates[blocked_pairs] = 0  # Assign 0 for pairs that are not used in training indexes

        pairs = torch.multinomial(
            candidates, num_samples=len(training_indexes), replacement=False, generator=self.generator
        ).tolist()

        if self.pairs is None:  # First call (initialization)
            self.pairs = pairs
            return None

        # Update pairs
        for index, pair in zip(training_indexes, pairs):
            self.pairs[index] = pair

        return None

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
