import typing

import pytorch_lightning as pl
import torchvision

from cyclegan import CycleGAN
from cyclegan.data.module import CycleGANDataModule


def train(
    dataset_type: typing.Literal["celeba", "custom"],
    dataset_root: str,
    feature: str = None,
    test_size: float = 0.3,
    image_size: int = 128,
    seed: int | None = 0,
    max_epochs: int = 3,
) -> None:
    datamodule = CycleGANDataModule(
        dataset_type=dataset_type,
        dataset_root=dataset_root,
        test_size=test_size,
        feature=feature,
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.Resize((image_size, image_size)), torchvision.transforms.ToTensor()]
        ),
        seed=seed,
    )

    model = CycleGAN()
    trainer = pl.Trainer(max_epochs=max_epochs)
    trainer.fit(model, datamodule)


def main() -> None:
    train(dataset_type="celeba", dataset_root="data", feature="Smiling")


if __name__ == "__main__":
    main()
