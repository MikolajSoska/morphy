import contextlib
import os
import pathlib
import tempfile

import numpy as np
import PIL.Image
import pytest

from cyclegan.data.dataset import CycleGANDataset


@contextlib.contextmanager
def mock_images(number_of_directories: int = 2, number_of_images: int = 3) -> str:
    """
    Creates temporary directories with images for test purposes.
    Images and directories will be automatically removed after usage.
    Images are created in format:
    `root_dir` -> `class_dir_n` -> `image_k.jpg`
    :param number_of_directories: Number of directories to create
    :param number_of_images: Number of images in each directory
    :return: path to root directory
    """
    with tempfile.TemporaryDirectory() as root_dir:
        for _ in range(number_of_directories):
            class_dir = tempfile.mkdtemp(dir=root_dir)
            for i in range(number_of_images):
                image = PIL.Image.fromarray(np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8))
                image.save(pathlib.Path(class_dir) / f"image_{i}.jpg")

        yield root_dir


@pytest.mark.parametrize("number_of_directories", [1, 2, 3])
@pytest.mark.parametrize("number_of_images", [1, 2, 3])
def test_mock_images(number_of_directories: int, number_of_images: int) -> None:
    with mock_images(number_of_directories=number_of_directories, number_of_images=number_of_images) as images_root:
        images_root = pathlib.Path(images_root)
        subdirectories = os.listdir(images_root)
        assert len(subdirectories) == number_of_directories
        for directory in subdirectories:
            images = os.listdir(images_root / directory)
            assert len(images) == number_of_images
            assert all(image_file.endswith(".jpg") for image_file in images)

    assert not images_root.exists()  # Check images deletion


def test_cyclegan_dataset() -> None:
    with mock_images(number_of_directories=2, number_of_images=3) as images_root:
        dataset = CycleGANDataset(images_root)
        assert len(dataset.classes) == 2
        assert len(dataset) == 6

        image, label = dataset[0]
        assert isinstance(image, PIL.Image.Image)
        assert isinstance(label, int)


def test_cyclegan_dataset_more_categories() -> None:
    with mock_images(number_of_directories=3) as images_root:
        with pytest.raises(AssertionError):
            CycleGANDataset(images_root)


def test_cyclegan_dataset_less_categories() -> None:
    with mock_images(number_of_directories=1) as images_root:
        with pytest.raises(AssertionError):
            CycleGANDataset(images_root)
