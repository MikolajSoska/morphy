import itertools
import typing

import pytorch_lightning as pl
import pytorch_lightning.loggers
import torch
import torch.functional
import torch.nn as nn
import torchmetrics

from .model.discriminator import Discriminator
from .model.generator import Generator


class CycleGAN(pl.LightningModule):
    """
    PytorchLightning module that is used for training and validation of the CycleGAN model.
    Module simultaneously trains two generators that are capable of transforming image from domain A into the domain B
    and vice-versa, together with discriminators for both domains.
    """

    TRAIN_PHASE_NAME = "train"
    VAL_PHASE_NAME = "val"

    # noinspection PyUnusedLocal
    # PytorchLightning saves hyperparameters from __init__ args, so this variable is used but not explicitly
    def __init__(
        self,
        in_channels: int = 3,
        residual_blocks: int = 6,
        learning_rate: float = 2e-4,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> None:
        """
        Parameters
        ----------
        in_channels : int, default: 3
            Number of channels in input images
        residual_blocks : int, default: 6
            Number of residual blocks in generators
        learning_rate : float, default: 2e-4
            Learning rate for both optimizers
        *args : typing.Any
            Additional module args
        **kwargs : typing.Any
            Additional module kwargs
        """
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False

        self.save_hyperparameters(logger=False)  # HParams are saved as config file separately
        self.generator_a = Generator(in_channels, residual_blocks)  # A -> B
        self.generator_b = Generator(in_channels, residual_blocks)  # B -> A
        self.discriminator_a = Discriminator(in_channels)  # Is A real
        self.discriminator_b = Discriminator(in_channels)  # Is B real

        self.fid = torchmetrics.image.FrechetInceptionDistance(feature=64, normalize=True)

    def __run_generators(
        self, image_a: torch.Tensor, image_b: torch.Tensor, phase: str = TRAIN_PHASE_NAME
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Method for running single generation step in training or validation.

        Parameters
        ----------
        image_a : torch.Tensor
            Image from the first domain
        image_b : torch.Tensor
            Image from the second domain

        Returns
        -------
        torch.Tensor
            Generator loss
        dict[str, torch.Tensor]
            Dictionary with generated images:
                - 'generated_a' - Transformed image B (B -> A)
                - 'generated_b' - Transformed image A (A -> B)
                - 'cycle_a' - Cycle image A (A -> B -> A)
                - 'cycle_b' - Cycle image B (B -> A -> B)
                - 'identity_a' - Identity image A (A -> A)
                - 'identity_b' - Identity image B (B -> B)
        """
        generated_a = self.generator_b(image_b)  # B (real) -> A (fake)
        generated_b = self.generator_a(image_a)  # A (real) -> B (fake)
        cycle_a = self.generator_b(generated_b)  # B (fake) -> A (fake)
        cycle_b = self.generator_a(generated_a)  # A (fake) -> B (fake)
        cycle_loss = nn.functional.l1_loss(cycle_a, image_a) + nn.functional.l1_loss(cycle_b, image_b)

        identity_a = self.generator_b(image_a)  # A (real) -> A (fake)
        identity_b = self.generator_a(image_b)  # B (real) -> B (fake)
        identity_loss = nn.functional.l1_loss(identity_a, image_a) + nn.functional.l1_loss(identity_b, image_b)

        generator_loss = cycle_loss + identity_loss

        self.fid.update(torch.cat((image_a, image_b), dim=0), real=True)
        self.fid.update(torch.cat((generated_a, generated_a), dim=0), real=False)

        self.log(f"{phase}_cycle_loss", cycle_loss)
        self.log(f"{phase}_identity_loss", identity_loss)
        self.log(f"{phase}_generator_loss", generator_loss, prog_bar=True)
        self.log(f"{phase}_FID", self.fid.compute())

        self.fid.reset()

        output = {
            "generated_a": generated_a,
            "generated_b": generated_b,
            "cycle_a": cycle_a,
            "cycle_b": cycle_b,
            "identity_a": identity_a,
            "identity_b": identity_b,
        }

        return generator_loss, output

    def __run_discriminator(
        self,
        real_a: torch.Tensor,
        real_b: torch.Tensor,
        fake_a: torch.Tensor,
        fake_b: torch.Tensor,
        phase: str = TRAIN_PHASE_NAME,
    ) -> torch.Tensor:
        """
        Method for running single discriminator step in training or validation.

        Parameters
        ----------
        real_a : torch.Tensor
            Image from the first domain
        real_b : torch.Tensor
            Image from the second domain
        fake_a : torch.Tensor
            Fake image from the first domain (B -> A)
        fake_b : torch.Tensor
            Fake image from the second domain (A -> B)

        Returns
        -------
        torch.Tensor
            Discriminator loss
        """
        prediction_fake_a = self.discriminator_a(fake_a)
        prediction_fake_b = self.discriminator_b(fake_b)
        prediction_real_a = self.discriminator_a(real_a)
        prediction_real_b = self.discriminator_b(real_b)

        target_fake = torch.zeros_like(prediction_fake_a)
        target_real = torch.ones_like(prediction_real_a)

        discriminator_a_real_loss = nn.functional.mse_loss(prediction_real_a, target_real)  # Real image A
        discriminator_a_fake_loss = nn.functional.mse_loss(prediction_fake_a, target_fake)  # Fake image A
        discriminator_b_real_loss = nn.functional.mse_loss(prediction_real_b, target_real)  # Real image B
        discriminator_b_fake_loss = nn.functional.mse_loss(prediction_fake_b, target_fake)  # Fake image B

        discriminator_loss = 0.5 * (
            discriminator_a_real_loss
            + discriminator_a_fake_loss
            + discriminator_b_real_loss
            + discriminator_b_fake_loss
        )

        self.log(f"{phase}_discriminator_a_real_loss", discriminator_a_real_loss)
        self.log(f"{phase}_discriminator_a_fake_loss", discriminator_a_fake_loss)
        self.log(f"{phase}_discriminator_b_real_loss", discriminator_b_real_loss)
        self.log(f"{phase}_discriminator_b_fake_loss", discriminator_b_fake_loss)
        self.log(f"{phase}_discriminator_a_loss", discriminator_a_real_loss + discriminator_a_fake_loss)
        self.log(f"{phase}_discriminator_b_loss", discriminator_b_real_loss + discriminator_b_fake_loss)
        self.log(f"{phase}_discriminator_loss", discriminator_loss, prog_bar=True)

        return discriminator_loss

    def __log_images(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        output_images: dict[str, torch.Tensor],
        step: int,
        phase: str,
    ) -> None:
        """
        Method logs images with the available loggers.

        Parameters
        ----------
        image_a : torch.Tensor
            Original image A
        image_b : torch.Tensor
            Original image B
        output_images : dict[str, torch.Tensor]
            Dictionary with generated images

        Returns
        -------
        None
        """

        def __log_images_single(
            _logger: pl.loggers.WandbLogger, real_image: torch.Tensor, from_image: str, to_image: str
        ) -> None:
            """
            Helper method for logging single image version (A or B), with proper descriptions.

            Parameters
            ----------
            _logger : pl.loggers.WandbLogger
                Logger instance
            real_image : torch.Tensor
                Real image
            from_image : str
                ID of starting image
            to_image : str
                ID of target image

            Returns
            -------
            None
            """
            from_image = from_image.upper()
            to_image = to_image.upper()
            _logger.log_image(
                images=[
                    real_image,
                    output_images[f"generated_{to_image.lower()}"],
                    output_images[f"cycle_{from_image.lower()}"],
                    output_images[f"identity_{from_image.lower()}"],
                ],
                key=f"Image {from_image} transformations ({phase})",
                step=step,
                caption=[
                    f"Image {from_image}",
                    f"Generated {to_image} ({from_image} -> {to_image})",
                    f"Cycle {from_image} ({from_image} -> {to_image} -> {from_image})",
                    f"Identity {from_image} ({from_image} -> {from_image})",
                ],
            )

        for logger in self.loggers:
            if isinstance(logger, pl.loggers.WandbLogger):
                __log_images_single(logger, image_a, from_image="A", to_image="B")
                __log_images_single(logger, image_b, from_image="B", to_image="A")

    def training_step(self, batch: list[torch.Tensor], batch_index: int) -> None:
        """
        Method defines single training step of CycleGAN.

        Parameters
        ----------
        batch : list[torch.Tensor]
            List of length = 2, that holds single batch of images from both domains
        batch_index : int
            Index number of given batch

        Returns
        -------
        None
        """
        image_a, image_b = batch

        optimizer_generator, optimizer_discriminator = self.optimizers()

        # Generators training
        self.toggle_optimizer(optimizer_generator)
        generator_loss, output_images = self.__run_generators(image_a, image_b, phase=self.TRAIN_PHASE_NAME)
        self.manual_backward(generator_loss)

        optimizer_generator.step()
        optimizer_generator.zero_grad()
        self.untoggle_optimizer(optimizer_generator)

        # noinspection PyUnresolvedReferences
        # PyChams says that trainer don't have a log_every_n_steps attribute
        if (batch_index + 1) % self.trainer.log_every_n_steps == 0:
            self.__log_images(image_a, image_b, output_images, step=batch_index, phase=self.TRAIN_PHASE_NAME)

        # Discriminators training
        self.toggle_optimizer(optimizer_discriminator)
        discriminator_loss = self.__run_discriminator(
            image_a,
            image_b,
            output_images["generated_a"].detach(),
            output_images["generated_b"].detach(),
            phase=self.TRAIN_PHASE_NAME,
        )
        self.manual_backward(discriminator_loss)

        optimizer_discriminator.step()
        optimizer_discriminator.zero_grad()
        self.untoggle_optimizer(optimizer_discriminator)

    def validation_step(self, batch: list[torch.Tensor], batch_index: int) -> None:
        """
        Method defines single validation step of CycleGAN.

        Parameters
        ----------
        batch : list[torch.Tensor]
            List of length = 2, that holds single batch of images from both domains
        batch_index : int
            Index number of given batch

        Returns
        -------
        None
        """
        image_a, image_b = batch
        _, output_images = self.__run_generators(image_a, image_b, phase=self.VAL_PHASE_NAME)
        # noinspection PyUnresolvedReferences
        # PyChams says that trainer don't have a log_every_n_steps attribute
        if (batch_index + 1) % self.trainer.log_every_n_steps == 0:
            self.__log_images(image_a, image_b, output_images, step=batch_index, phase=self.VAL_PHASE_NAME)

        self.__run_discriminator(
            image_a, image_b, output_images["generated_a"], output_images["generated_b"], phase=self.VAL_PHASE_NAME
        )

    def forward(
        self, image_a: torch.Tensor = None, image_b: torch.Tensor = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert image_a is not None or image_b is not None, "At least one image must be provided."

        result = []

        if image_a is not None:
            result.append(self.generator_a(image_a))  # A (real) -> B (fake)

        if image_b is not None:
            result.append(self.generator_b(image_b))  # B (real) -> A (fake)

        return tuple(result) if len(result) > 1 else result[0]

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        """
        Method creates optimizers both generators and discriminators.

        Returns
        -------
        list[torch.optim.Optimizer]
            List of length = 2, that holds generator optimizer and discriminator optimizer
        """
        optimizer_generators = torch.optim.Adam(
            itertools.chain(self.generator_a.parameters(), self.generator_b.parameters()),
            lr=self.hparams.learning_rate,
        )
        optimizer_discriminators = torch.optim.Adam(
            itertools.chain(self.discriminator_a.parameters(), self.discriminator_b.parameters()),
            lr=self.hparams.learning_rate,
        )

        return [optimizer_generators, optimizer_discriminators]
