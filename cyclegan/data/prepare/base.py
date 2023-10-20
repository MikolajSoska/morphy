import abc
import pathlib
import typing


class Preparer(abc.ABC):
    """Base abstract class for classes that prepares datasets for CycleGAN format"""

    @classmethod
    @abc.abstractmethod
    def prepare_pairs(cls, *args: typing.Any, **kwargs: typing.Any) -> str | pathlib.Path:
        """
        Method should prepare dataset pairs that will be used in training,
        and return directory where the images are stored.

        Parameters
        ----------
        *args : typing.Any
            Method positional args
        **kwargs : typing.Any
            Method keyword args

        Returns
        -------
        str | pathlib.Path
            Path where the images are stored
        """
        pass

    @classmethod
    @abc.abstractmethod
    def get_target_dir(cls, root_dir: str | pathlib.Path, **kwargs: typing.Any) -> pathlib.Path:
        """
        Method constructs path to the directory with the dataset,
        based on the passed root directory and other optional features.

        Parameters
        ----------
        root_dir : str | pathlib.Path
            Root directory for the dataset images.
        **kwargs : typing.Any
            Method keyword args

        Returns
        -------
        pathlib.Path
            Constructed path
        """
        pass
