import pathlib
import typing

import pytorch_lightning as pl
import torch
import torch.utils.data

from .dataset import CycleGANDataset
from .loader import CycleGANDataLoader
from .prepare.celeba import CelebAPreparer


class CycleGANDataModule(pl.LightningDataModule):
    """
    Lightning DataModule class that stores are data for CycleGAN training and testing.
    Custom datasets must be created before initialization of this module.
    Public datasets from the web (like CelebA) are downloaded automatically during setup phase.
    """

    def __init__(
        self,
        dataset_type: typing.Literal["custom", "celeba"],
        dataset_root: str | pathlib.Path,
        test_size: float,
        feature: str = None,
        batch_size: int = 1,
        transform: typing.Callable = None,
        seed: int = None,
    ) -> None:
        """
        Parameters
        ----------
        dataset_type : typing.Literal["custom", "celeba"]
            Type of the dataset. "Custom" refers to any dataset created manually.
             CelebA is a public dataset downloaded from the web.
        dataset_root : str | pathlib.Path
            Root directory of the dataset
        test_size : float
            Test size fraction value. Must be between 0 and 1
        feature : str, default: None
            Feature name used in the CelebA dataset. Will determine how images pairs will be created.
        batch_size : int, default: 1
            Data batch size
        transform : typing.Callable, default: None
            Transforms for the sampled images
        seed : int, default: None
            Random seed
        """
        assert 0 < test_size < 1, f"'test_size' must be a value between 0 and 1. Passed: {test_size}."
        assert feature is not None or dataset_type == "custom", "'feature' must be passed when using CelebA dataset."

        super().__init__()
        self.dataset_type = dataset_type
        self.dataset_root = dataset_root
        self.feature = feature
        self.test_size = test_size
        self.batch_size = batch_size
        self.transform = transform
        self.seed = seed

        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None
        self.dataset_predict = None

    def prepare_data(self) -> None:
        """
        Datasets need download and preparation (like CelebA), are processed in this method.

        Returns
        -------
        None
        """
        if self.dataset_type == "celeba":
            CelebAPreparer.prepare_pairs(
                images_root=self.dataset_root, feature=self.feature, target_root=self.dataset_root
            )

    def setup(self, stage: str) -> None:
        """
        Method initializes CycleGANDataset instances for the stage. In this situation all phases use same datasets.

        Parameters
        ----------
        stage : str
            Experiment phase

        Returns
        -------
        None
        """
        if self.dataset_type == "celeba":
            data_dir = CelebAPreparer.get_target_dir(self.dataset_root, self.feature)
        else:
            data_dir = self.dataset_root

        dataset = CycleGANDataset(images_root=data_dir, transform=self.transform, seed=self.seed)
        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(self.seed)

        self.dataset_train, self.dataset_val = torch.utils.data.random_split(
            dataset=dataset,
            lengths=(1 - self.test_size, self.test_size),
            generator=generator,
        )

        # There is no separate datasets for test and predict phase, so validation split is used again here
        self.dataset_test = self.dataset_val
        self.dataset_predict = self.dataset_val

    def train_dataloader(self) -> CycleGANDataLoader:
        """
        Method initializes training dataloader.

        Returns
        -------
        CycleGANDataLoader
            Initialized dataloader
        """
        return CycleGANDataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, training=True)

    def val_dataloader(self) -> CycleGANDataLoader:
        """
        Method initializes validation dataloader.

        Returns
        -------
        CycleGANDataLoader
            Initialized dataloader
        """
        return CycleGANDataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self) -> CycleGANDataLoader:
        """
        Method initializes testing dataloader.

        Returns
        -------
        CycleGANDataLoader
            Initialized dataloader
        """
        return CycleGANDataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self) -> CycleGANDataLoader:
        """
        Method initializes predict dataloader.

        Returns
        -------
        CycleGANDataLoader
            Initialized dataloader
        """
        return CycleGANDataLoader(self.dataset_predict, batch_size=self.batch_size, shuffle=False)
