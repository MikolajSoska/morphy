"""Module contains various neural network layers used in CycleGAN model"""
import typing

import torch
import torch.nn as nn


class ConvolutionBlock(nn.Module):
    """
    ConvolutionBlock layer with structure Convolution -> InstanceNorm -> Activation
    """

    def __init__(
        self,
        filters_in: int,
        filters_out: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        output_padding: int | tuple[int, int] = 0,
        activation: typing.Literal["relu", "tanh", "none"] = "relu",
        convolution_type: typing.Literal["base", "transpose"] = "base",
    ) -> None:
        """
        :param filters_in: Number of convolution input filters
        :param filters_out: Number of convolution output filters
        :param kernel_size: Size of convolution kernel
        :param stride: Convolution stride
        :param padding: Convolutions padding size
        :param output_padding Convolution output padding (used only when `convolution_type` = "transpose")
        :param activation: Activation type, used after norm layer
        :param convolution_type: Which type of convolution use (nn.Conv2d or nn.ConvTranspose2d)
        """
        super().__init__()
        match convolution_type:
            case "base":
                convolution_class = nn.Conv2d
                class_params = {}
            case "transpose":
                convolution_class = nn.ConvTranspose2d
                class_params = {"output_padding": output_padding}
            case _:
                raise ValueError(f"Invalid {convolution_type = }")

        layers = [
            convolution_class(
                in_channels=filters_in,
                out_channels=filters_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                **class_params,
            ),
            nn.InstanceNorm2d(filters_out),
        ]

        match activation:
            case "relu":
                activation_class = nn.ReLU
            case "tanh":
                activation_class = nn.Tanh
            case "none":
                activation_class = None
            case _:
                raise ValueError(f"Invalid {activation = }")

        if activation_class is not None:
            layers.append(activation_class())

        self.block = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: Input tensor with shape batch x `filters_in` x height x width
        :return: Processed tensor with shape batch x `filters_out` x height_out x width_out
        """
        return self.block(inputs)


class ResidualLayer(nn.Module):
    """
    Class wraps layer `f` and processes input `x_in` like: `x_out = f(x_in) + x_in`
    """

    def __init__(self, module: nn.Module) -> None:
        """
        :param module: Module to be wrapped
        """
        super().__init__()
        self.module = module

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: Input tensor for residual layer
        :return: Output tensor after residual layer
        """
        return self.module(inputs) + inputs
