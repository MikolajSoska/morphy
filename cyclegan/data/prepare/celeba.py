import os
import pathlib
import typing

import numpy as np
import torchvision
import tqdm

from .base import Preparer


class CelebAPreparer(Preparer):
    """Class prepares CelebA dataset for CycleGAN training."""

    IMAGES_DIRECTORY = "img_align_celeba"

    @classmethod
    def prepare_pairs(
        cls, images_root: str | pathlib.Path, feature: str, target_root: str | pathlib.Path
    ) -> str | pathlib.Path:
        """
        Method constructs image pairs from CelebA dataset by given feature.

        Parameters
        ----------
        images_root : str | pathlib.Path
            Root directory where the CelebA dataset will be stored
        feature : str
            Name of the feature from CelebA dataset
        target_root : str | pathlib.Path
            Root directory where prepared pairs will be stored

        Returns
        -------
        str | pathlib.Path
            Directory with images that can be used in the dataset construction.
        """
        if isinstance(images_root, str):
            images_root = pathlib.Path(images_root)
        if isinstance(target_root, str):
            target_root = pathlib.Path(target_root)

        dataset = torchvision.datasets.CelebA(root=str(images_root), download=True)
        if feature not in dataset.attr_names:
            raise ValueError(f"Invalid {feature = }. Available values: {', '.join(dataset.attr_names)}.")

        feature_index = dataset.attr_names.index(feature)
        filenames = np.asarray(dataset.filename)  # Convert to ndarray for easier indexing
        positive_pairs = filenames[dataset.attr[:, feature_index] == 1]
        negative_pairs = filenames[dataset.attr[:, feature_index] != 1]

        cls.__create_images_directory(
            dataset_root=images_root / dataset.base_folder,
            save_dir=target_root / feature / "positive",
            images=positive_pairs,
        )
        cls.__create_images_directory(
            dataset_root=images_root / dataset.base_folder,
            save_dir=target_root / feature / "negative",
            images=negative_pairs,
        )

        return target_root / feature

    @classmethod
    def __create_images_directory(
        cls, dataset_root: pathlib.Path, save_dir: pathlib.Path, images: typing.Iterable
    ) -> None:
        """
        Method constructs images directory with symlinks to the original data.

        Parameters
        ----------
        dataset_root : pathlib.Path
            Root directory where CelebA dataset is stored
        save_dir : pathlib.Path
            Directory when the images will be saves
        images : typing.Iterable
            Sequence of images names

        Returns
        -------
        None
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        images_dir = (dataset_root / cls.IMAGES_DIRECTORY).resolve()  # Symlinks need absolute path
        for image in tqdm.tqdm(images, desc=f"Preparing images in {save_dir}"):
            os.symlink(images_dir / image, save_dir / image)
