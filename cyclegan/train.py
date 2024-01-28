import os
import pathlib

import dotenv

# noinspection PyPackageRequirements
# This package is installed as hydra-core, PyCharm doesn't recognize this
import hydra
import omegaconf
import pytorch_lightning as pl
import pytorch_lightning.loggers
import torchvision
import wandb

from cyclegan import CycleGAN, utils
from cyclegan.data.module import CycleGANDataModule


@hydra.main(config_path="../configs", config_name="train.yaml", version_base=None)
def train(config: omegaconf.DictConfig) -> None:
    """
    Runs CycleGAN training with parameters from config.

    Parameters
    ----------
    config : omegaconf.DictConfig
        Training config read from YAML file

    Returns
    -------
    None
    """
    dotenv.load_dotenv(dotenv.find_dotenv())
    wandb.login(key=os.environ["WANDB_API_KEY"])

    csv_logger = pl.loggers.CSVLogger(save_dir=config.training.logs_dir, name=config.training.experiment_name)
    wandb_logger = pl.loggers.WandbLogger(
        project="morphy",
        save_dir=config.training.logs_dir,
        name=f"{config.training.experiment_name}/version_{csv_logger.version}",
    )
    csv_logger.log_hyperparams(config)
    wandb_logger.log_hyperparams(utils.flatten_dict(omegaconf.OmegaConf.to_object(config)))

    datamodule = CycleGANDataModule(
        dataset_type=config.dataset.type,
        dataset_root=config.dataset.root_dir,
        test_size=config.dataset.test_size,
        feature=config.dataset.feature,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((config.dataset.image_size, config.dataset.image_size)),
                torchvision.transforms.ToTensor(),
            ]
        ),
        seed=config.training.seed,
    )

    model = CycleGAN(
        residual_blocks=config.model.residual_blocks,
        learning_rate=config.training.learning_rate,
    )
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        logger=[csv_logger, wandb_logger],
    )
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()
