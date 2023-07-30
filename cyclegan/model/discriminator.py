import torch
import torch.nn as nn

from .layers import ConvolutionBlock


class Discriminator(nn.Module):
    """CycleGAN discriminator that checks if image is original or generated"""

    def __init__(self, in_channels: int = 3) -> None:
        """
        Parameters
        ----------
        in_channels : int, default: 3
            Number of input channels in data
        """
        super().__init__()
        self.network = nn.Sequential(
            ConvolutionBlock(
                filters_in=in_channels, filters_out=64, kernel_size=4, stride=2, padding=1, relu_slope=0.2, norm="none"
            ),
            ConvolutionBlock(filters_in=64, filters_out=128, kernel_size=4, stride=2, padding=1, relu_slope=0.2),
            ConvolutionBlock(filters_in=128, filters_out=256, kernel_size=4, stride=2, padding=1, relu_slope=0.2),
            ConvolutionBlock(filters_in=256, filters_out=512, kernel_size=4, padding=1, relu_slope=0.2),
            ConvolutionBlock(filters_in=512, filters_out=1, kernel_size=4, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward method of CycleGAN discriminator.

        Parameters
        ----------
        inputs : torch.Tensor
            Input image tensor with shape batch x channels x height x width

        Returns
        -------
        torch.Tensor
            Tensor prediction if image is original of generated in the shape batch x 1
        """
        return self.network(inputs)
