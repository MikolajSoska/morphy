import typing

import torch.utils.data

from .dataset import CycleGANDataset


class CycleGANDataLoader(torch.utils.data.DataLoader):
    """Custom dataloader for CycleGAN data"""

    def __init__(
        self,
        dataset: CycleGANDataset | torch.utils.data.Subset[CycleGANDataset],
        training: bool = False,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> None:
        """
        Make sure that dataset has correct type.

        Parameters
        ----------
        dataset : CycleGANDataset | torch.utils.data.Subset[CycleGANDataset]
            An instance of CycleGANDataset or a subset with CycleGANDataset
        training : bool, default: False
            Is this loader used for model training
        *args : typing.Any
            DataLoader args
        **kwargs : typing.Any
            DataLoader kwargs
        """
        actual_dataset = dataset.dataset if (is_subset := isinstance(dataset, torch.utils.data.Subset)) else dataset

        if not isinstance(actual_dataset, CycleGANDataset):
            raise ValueError(f"Only instances of `CycleGANDataset` can be used in this loader. {type(dataset)}")

        super().__init__(dataset, *args, **kwargs)
        self.actual_dataset = actual_dataset
        self.training = training
        self.is_subset = is_subset

    def __iter__(self) -> typing.Any:
        """Override this method to reset pairs at each training epoch start"""
        if self.training:
            if self.is_subset:
                assert isinstance(self.dataset, torch.utils.data.Subset)  # Add assert to supress PyCharm warnings
                self.actual_dataset.reset_pairs(self.dataset.indices)
            else:
                self.actual_dataset.reset_pairs()

        return super().__iter__()
