import typing

import pytest
import torch.nn as nn

import cyclegan.model.layers as layers


def __test_convolution_block_init(
    conv_block: layers.ConvolutionBlock, layers_types: typing.Sequence[typing.Type]
) -> None:
    assert len(conv_block.block) == len(layers_types)
    for layer, layer_type in zip(conv_block.block, layers_types):
        assert isinstance(layer, layer_type)


def test_convolution_block_init_default() -> None:
    in_channels = 4
    out_channels = 8
    conv_block = layers.ConvolutionBlock(filters_in=in_channels, filters_out=out_channels)
    __test_convolution_block_init(conv_block, [nn.Conv2d, nn.InstanceNorm2d, nn.ReLU])

    assert conv_block.block[0].in_channels == in_channels
    assert conv_block.block[0].out_channels == out_channels

    # Check default values
    assert conv_block.block[0].kernel_size == (3, 3)
    assert conv_block.block[0].stride == (1, 1)
    assert conv_block.block[0].padding == (0, 0)


@pytest.mark.parametrize("value", [1, 3, (3, 3), (1, 2)])
def test_convolution_block_init_params(value: int | tuple[int, int]) -> None:
    conv_block = layers.ConvolutionBlock(filters_in=4, filters_out=8, kernel_size=value, stride=value, padding=value)
    __test_convolution_block_init(conv_block, [nn.Conv2d, nn.InstanceNorm2d, nn.ReLU])

    if isinstance(value, int):
        value = (value, value)

    # Check default values
    assert conv_block.block[0].kernel_size == value
    assert conv_block.block[0].stride == value
    assert conv_block.block[0].padding == value


@pytest.mark.parametrize(["conv_type", "conv_class"], [("base", nn.Conv2d), ("transpose", nn.ConvTranspose2d)])
@pytest.mark.parametrize(["norm_type", "norm_class"], [("instance", nn.InstanceNorm2d), ("none", None)])
@pytest.mark.parametrize(
    ["activation_type", "activation_class"], [("relu", nn.ReLU), ("tanh", nn.Tanh), ("none", None)]
)
def test_convolution_block_init_types(
    conv_type: str,
    conv_class: typing.Type,
    norm_type: str,
    norm_class: typing.Type,
    activation_type: str,
    activation_class: typing.Type,
) -> None:
    # noinspection PyTypeChecker
    # Disable warning about string passes to literal types
    conv_block = layers.ConvolutionBlock(
        filters_in=4, filters_out=8, convolution_type=conv_type, activation=activation_type, norm=norm_type
    )
    layers_types = [layer_type for layer_type in [conv_class, norm_class, activation_class] if layer_type is not None]
    __test_convolution_block_init(conv_block, layers_types)


# noinspection PyTypeChecker
# Same as in previous method
def test_convolution_block_init_types_wrong() -> None:
    wrong_type = "wrong type"
    with pytest.raises(ValueError):
        layers.ConvolutionBlock(filters_in=4, filters_out=8, convolution_type=wrong_type)

    with pytest.raises(ValueError):
        layers.ConvolutionBlock(filters_in=4, filters_out=8, norm=wrong_type)

    with pytest.raises(ValueError):
        layers.ConvolutionBlock(filters_in=4, filters_out=8, activation=wrong_type)


@pytest.mark.parametrize("output_padding", [0, 1, (1, 2), (3, 3)])
def test_convolution_block_init_output_padding(output_padding: int | tuple[int, int]) -> None:
    conv_block = layers.ConvolutionBlock(
        filters_in=4, filters_out=8, convolution_type="transpose", output_padding=output_padding
    )
    __test_convolution_block_init(conv_block, [nn.ConvTranspose2d, nn.InstanceNorm2d, nn.ReLU])
    if isinstance(output_padding, int):
        output_padding = (output_padding, output_padding)

    assert conv_block.block[0].output_padding == output_padding


@pytest.mark.parametrize(["relu_slope", "expected_class"], [(0, nn.ReLU), (0.2, nn.LeakyReLU)])
def test_convolution_block_init_leaky_relu(relu_slope: float, expected_class: typing.Type) -> None:
    conv_block_relu_activation = layers.ConvolutionBlock(
        filters_in=4, filters_out=8, activation="relu", relu_slope=relu_slope
    )
    __test_convolution_block_init(conv_block_relu_activation, [nn.Conv2d, nn.InstanceNorm2d, expected_class])

    if expected_class == nn.LeakyReLU:
        assert conv_block_relu_activation.block[2].negative_slope == relu_slope
