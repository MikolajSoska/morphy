"""Module contains Generator class from CycleGAN model"""

import torch
import torch.nn as nn

from .layers import ConvolutionBlock, ResidualLayer


class Generator(nn.Module):
    """
    CycleGAN generator that transforms image from one domain to other
    """

    def __init__(self, in_channels: int = 3, residual_blocks: int = 6) -> None:
        """
        :param in_channels: Number of channels in the input image
        :param residual_blocks: number of residual blocks used in the generator
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.ReflectionPad2d(padding=3),
            ConvolutionBlock(filters_in=in_channels, filters_out=64, kernel_size=7),
            ConvolutionBlock(filters_in=64, filters_out=128, stride=2, padding=1),
            ConvolutionBlock(filters_in=128, filters_out=256, stride=2, padding=1),
            *[
                ResidualLayer(
                    nn.Sequential(
                        nn.ReflectionPad2d(padding=1),
                        ConvolutionBlock(filters_in=256, filters_out=256),
                        nn.ReflectionPad2d(padding=1),
                        ConvolutionBlock(filters_in=256, filters_out=256, activation="none"),
                    )
                )
                for _ in range(residual_blocks)
            ],
            ConvolutionBlock(
                filters_in=256, filters_out=128, stride=2, padding=1, output_padding=1, convolution_type="transpose"
            ),
            ConvolutionBlock(
                filters_in=128, filters_out=64, stride=2, padding=1, output_padding=1, convolution_type="transpose"
            ),
            nn.ReflectionPad2d(padding=3),
            ConvolutionBlock(filters_in=64, filters_out=in_channels, kernel_size=7, activation="tanh"),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward method of CycleGAN generator. Transformers image tensor into other image tensor.
        :param inputs: input image tensor with shape batch x channels x height x width
        :return: Output image tensor with shape batch x channels x height x width
        """
        return self.network(inputs)
