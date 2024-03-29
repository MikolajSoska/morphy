import os
import pathlib
import shutil
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
        cls,
        images_root: str | pathlib.Path,
        feature: str,
        target_root: str | pathlib.Path,
        use_symlinks: bool = False,
        override_files: bool = False,
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
        use_symlinks : bool, default: False
            When true will create symlinks instead of copying files when creating pairs.
            Symlinks will not use additional memory, but will be slower during read
        override_files : bool, default: False
            Whether to override existing files or to skip them

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
            use_symlinks=use_symlinks,
            override=override_files,
        )
        cls.__create_images_directory(
            dataset_root=images_root / dataset.base_folder,
            save_dir=target_root / feature / "negative",
            images=negative_pairs,
            use_symlinks=use_symlinks,
            override=override_files,
        )

        return cls.get_target_dir(target_root, feature)

    @classmethod
    def get_target_dir(cls, root_dir: str | pathlib.Path, feature: str = None) -> pathlib.Path:
        """
        Method creates path to the save directory for CelebA images, based on thr feature name.
        Path is constructed as: root_dir / feature

        Parameters
        ----------
        root_dir : str | pathlib.Path
            Dataset root directory
        feature : str
            CelebA feature name

        Returns
        -------
        pathlib.Path
            Constructed path
        """
        assert feature is not None, "'feature' parameter must be passed to this method."

        if isinstance(root_dir, str):
            root_dir = pathlib.Path(root_dir)

        return root_dir / feature

    @classmethod
    def __create_images_directory(
        cls,
        dataset_root: pathlib.Path,
        save_dir: pathlib.Path,
        images: typing.Iterable,
        use_symlinks: bool,
        override: bool,
    ) -> None:
        """
        Method constructs images directory.

        Parameters
        ----------
        dataset_root : pathlib.Path
            Root directory where CelebA dataset is stored
        save_dir : pathlib.Path
            Directory when the images will be saves
        images : typing.Iterable
            Sequence of images names
        use_symlinks : bool
            Whether to use symlinks or copy files
        override : bool
            Whether to override existing files or to skip them

        Returns
        -------
        None
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        images_dir = (dataset_root / cls.IMAGES_DIRECTORY).resolve()  # Symlinks need absolute path
        save_method = os.symlink if use_symlinks else shutil.copy

        images_to_create = set(images) - set(os.listdir(save_dir)) if not override else images
        if len(images_to_create) == 0:
            print(f"Images in {save_dir} are already created.")
            return

        for image in tqdm.tqdm(images_to_create, desc=f"Preparing images in {save_dir}"):
            save_method(images_dir / image, save_dir / image)
