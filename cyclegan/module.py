import itertools
import typing

import pytorch_lightning as pl
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

        self.save_hyperparameters()
        self.generator_a = Generator(in_channels, residual_blocks)  # A -> B
        self.generator_b = Generator(in_channels, residual_blocks)  # B -> A
        self.discriminator_a = Discriminator(in_channels)  # Is A real
        self.discriminator_b = Discriminator(in_channels)  # Is B real

        self.fid = torchmetrics.image.FrechetInceptionDistance(feature=64, normalize=True)

    def __run_generators(
        self, image_a: torch.Tensor, image_b: torch.Tensor, phase: str = "train"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        torch.Tensor
            Transformed image B (B -> A)
        torch.Tensor
            Transformed image A (A -> B)
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
        return generator_loss, generated_a, generated_b

    def __run_discriminator(
        self,
        real_a: torch.Tensor,
        real_b: torch.Tensor,
        fake_a: torch.Tensor,
        fake_b: torch.Tensor,
        phase: str = "train",
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
        generator_loss, generated_a, generated_b = self.__run_generators(image_a, image_b, phase="train")
        self.manual_backward(generator_loss)

        optimizer_generator.step()
        optimizer_generator.zero_grad()
        self.untoggle_optimizer(optimizer_generator)

        # Discriminators training
        self.toggle_optimizer(optimizer_discriminator)
        discriminator_loss = self.__run_discriminator(
            image_a, image_b, generated_a.detach(), generated_b.detach(), phase="train"
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
        _, generated_a, generated_b = self.__run_generators(image_a, image_b, phase="val")
        self.__run_discriminator(image_a, image_b, generated_a, generated_b, phase="val")

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
