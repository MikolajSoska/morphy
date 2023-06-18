import os
import pathlib

import torchvision


class CycleGANDataset(torchvision.datasets.ImageFolder):
    def __init__(self, images_root: str | pathlib.Path) -> None:
        assert (
            len(os.listdir(images_root)) == 2
        ), "Images root should have exactly 2 subdirectories, that will be used as CycleGAN domains"

        super().__init__(images_root)
