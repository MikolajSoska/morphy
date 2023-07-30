import collections
import contextlib
import os
import pathlib
import tempfile
import typing

import numpy as np
import PIL.Image
import pytest
import torch
import torchvision

from cyclegan.data.dataset import CycleGANDataset


@contextlib.contextmanager
def mock_images(number_of_directories: int = 2, number_of_images: int | typing.Sequence[int] = 3) -> str:
    """
    Creates temporary directories with images for test purposes.
    Images and directories will be automatically removed after usage.
    Images are created in format:
        `root_dir` -> `class_dir_n` -> `image_k.jpg`

    Parameters
    ----------
    number_of_directories : int, default: 2
        Number of directories to create
    number_of_images : int | typing.Sequence[int], default: 3
        Number of images in each directory

    Returns
    -------
    str
        Path to root directory
    """
    if isinstance(number_of_images, int):
        number_of_images = [number_of_images] * number_of_directories
    assert number_of_directories == len(number_of_images), (
        "If passing `number_of_images` as list it length " "should be equal to `number_of_directories`"
    )

    with tempfile.TemporaryDirectory() as root_dir:
        for directory_images in number_of_images:
            class_dir = tempfile.mkdtemp(dir=root_dir)
            for i in range(directory_images):
                image = PIL.Image.fromarray(np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8))
                image.save(pathlib.Path(class_dir) / f"image_{i}.jpg")

        yield root_dir


def load_pil_image(filename: str) -> PIL.Image.Image:
    """
    Default loader in dataset calls .convert() method from PIL on image, which deletes `filename` attribute.
    This attribute is needed for some tests, so using custom loader here

    Parameters
    ----------
    filename : str
        Image filename

    Returns
    -------
    PIL.Image.Image
        Opened image
    """
    return PIL.Image.open(filename)


@pytest.mark.parametrize("number_of_directories", [1, 2, 3])
@pytest.mark.parametrize("number_of_images", [1, 2, 3])
def test_mock_images(number_of_directories: int, number_of_images: int) -> None:
    with mock_images(number_of_directories=number_of_directories, number_of_images=number_of_images) as images_root:
        images_root = pathlib.Path(images_root)
        subdirectories = os.listdir(images_root)
        assert len(subdirectories) == number_of_directories
        if isinstance(number_of_images, int):
            number_of_images = [number_of_images] * number_of_directories

        for directory, directory_images in zip(subdirectories, number_of_images):
            images = os.listdir(images_root / directory)
            assert len(images) == directory_images
            assert all(image_file.endswith(".jpg") for image_file in images)

    assert not images_root.exists()  # Check images deletion


@pytest.mark.parametrize(["number_of_directories", "number_of_images"], [(2, [2, 3]), (2, 2), (3, [4, 5, 5])])
def test_mock_images_different_sizes(number_of_directories: int, number_of_images: int | list[int]) -> None:
    with mock_images(number_of_directories=number_of_directories, number_of_images=number_of_images) as images_root:
        images_root = pathlib.Path(images_root)
        subdirectories = os.listdir(images_root)
        assert len(subdirectories) == number_of_directories
        if isinstance(number_of_images, int):
            number_of_images = [number_of_images] * number_of_directories

        real_number_of_images = []
        for directory in subdirectories:
            images = os.listdir(images_root / directory)
            assert all(image_file.endswith(".jpg") for image_file in images)
            real_number_of_images.append(len(images))

        # Fastest way to compare two unordered lists
        assert collections.Counter(real_number_of_images) == collections.Counter(number_of_images)

    assert not images_root.exists()  # Check images deletion


def test_cyclegan_dataset_more_categories() -> None:
    with mock_images(number_of_directories=3) as images_root:
        with pytest.raises(AssertionError):
            CycleGANDataset(images_root)


def test_cyclegan_dataset_less_categories() -> None:
    with mock_images(number_of_directories=1) as images_root:
        with pytest.raises(AssertionError):
            CycleGANDataset(images_root)


@pytest.mark.parametrize("number_of_images", [3, [2, 3], [3, 2], [8, 8], [3, 7]])
def test_cyclegan_dataset_elements_numbers(number_of_images: int | list[int]) -> None:
    number_of_directories = 2
    with mock_images(number_of_directories=number_of_directories, number_of_images=number_of_images) as images_root:
        if isinstance(number_of_images, int):
            number_of_images = [number_of_images] * number_of_directories

        dataset = CycleGANDataset(images_root)
        assert len(dataset.classes) == 2
        assert len(dataset.imgs) == sum(number_of_images)
        assert len(dataset) == min(number_of_images)
        assert len(dataset.first_class) == min(number_of_images)
        assert len(dataset.second_class) == max(number_of_images)


def test_cyclegan_dataset_no_transforms() -> None:
    with mock_images(number_of_directories=2, number_of_images=3) as images_root:
        dataset = CycleGANDataset(images_root)
        first_image, second_image = dataset[0]
        assert isinstance(first_image, PIL.Image.Image)
        assert isinstance(second_image, PIL.Image.Image)


def test_cyclegan_dataset_transforms() -> None:
    with mock_images(number_of_directories=2, number_of_images=3) as images_root:
        dataset = CycleGANDataset(images_root, transform=torchvision.transforms.ToTensor())
        first_image, second_image = dataset[0]
        assert isinstance(first_image, torch.Tensor)
        assert isinstance(second_image, torch.Tensor)


def test_cyclegan_dataset_no_seed() -> None:
    with mock_images(number_of_directories=2, number_of_images=100) as images_root:
        first_dataset = CycleGANDataset(images_root, loader=load_pil_image)
        second_dataset = CycleGANDataset(images_root, loader=load_pil_image)

        first_samples = {sample.filename for sample in first_dataset[0]}
        second_samples = {sample.filename for sample in second_dataset[0]}
        assert first_samples != second_samples


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cyclegan_dataset_seed(seed: int) -> None:
    with mock_images(number_of_directories=2, number_of_images=3) as images_root:
        first_dataset = CycleGANDataset(images_root, seed=seed, loader=load_pil_image)
        second_dataset = CycleGANDataset(images_root, seed=seed, loader=load_pil_image)

        def check_images_similarity() -> None:
            first_samples = [image.filename for sample in first_dataset for image in sample]
            second_samples = [image.filename for sample in second_dataset for image in sample]
            assert first_samples == second_samples

        check_images_similarity()
        first_dataset.reset_pairs()
        second_dataset.reset_pairs()
        check_images_similarity()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cyclegan_dataset_different_seeds(seed: int) -> None:
    with mock_images(number_of_directories=2, number_of_images=3) as images_root:
        first_dataset = CycleGANDataset(images_root, seed=seed, loader=load_pil_image)
        second_dataset = CycleGANDataset(images_root, seed=seed + 1, loader=load_pil_image)

        first_samples = [image.filename for sample in first_dataset for image in sample]
        second_samples = [image.filename for sample in second_dataset for image in sample]
        assert first_samples != second_samples


def test_cyclegan_dataset_reset_pairs() -> None:
    with mock_images(number_of_directories=2, number_of_images=3) as images_root:
        dataset = CycleGANDataset(images_root, loader=load_pil_image)

        samples = [image.filename for sample in dataset for image in sample]
        same_samples = [image.filename for sample in dataset for image in sample]
        assert samples == same_samples

        dataset.reset_pairs()
        different_samples = [image.filename for sample in dataset for image in sample]
        assert samples != different_samples


@pytest.mark.parametrize("number_of_images", [3, [2, 3], [3, 2], [8, 8], [3, 7]])
def test_cyclegan_dataset_samples_uniqueness(number_of_images: int | list[int]) -> None:
    with mock_images(number_of_directories=2, number_of_images=number_of_images) as images_root:
        dataset = CycleGANDataset(images_root, loader=load_pil_image)

        first_class_samples, second_class_samples = tuple(
            zip(*[(first.filename, second.filename) for (first, second) in dataset])
        )
        assert len(first_class_samples) == len(set(first_class_samples))
        assert len(second_class_samples) == len(set(second_class_samples))
        assert len(set(first_class_samples) & set(second_class_samples)) == 0
