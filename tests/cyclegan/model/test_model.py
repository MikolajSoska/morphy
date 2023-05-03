"""
General test for PyTorch modules
"""
import itertools
import random

import pytest
import torch
import torch.nn as nn

from cyclegan.model import Discriminator, Generator


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("model_cpu", [Generator(), Discriminator()])
@torch.inference_mode()
def test_device_moving(model_cpu: nn.Module) -> None:
    inputs_cpu = torch.testing.make_tensor((8, 3, 256, 256), dtype=torch.float, device="cpu")
    outputs_cpu = model_cpu(inputs_cpu)

    model_cuda = model_cpu.to("cuda")
    inputs_cuda = inputs_cpu.to("cuda")
    outputs_cuda = model_cuda(inputs_cuda)

    model_back_on_cpu = model_cuda.to("cpu")
    inputs_back_on_cpu = inputs_cuda.to("cpu")
    outputs_back_on_cpu = model_back_on_cpu(inputs_back_on_cpu)

    torch.testing.assert_close(outputs_cpu, outputs_cuda.cpu(), rtol=0.001, atol=0.001)  # Need larger tolerance
    torch.testing.assert_close(outputs_cpu, outputs_back_on_cpu)


@pytest.mark.parametrize("model", [Generator(), Discriminator()])
def test_all_parameters_updated(model: nn.Module) -> None:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = torch.testing.make_tensor((8, 3, 256, 256), dtype=torch.float, device=device)

    model = model.to(device)
    outputs = model(inputs)

    loss = outputs.mean()
    loss.backward()
    optimizer.step()

    for param_name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, param_name
            assert torch.sum(param.grad**2) != 0, param


@pytest.mark.parametrize("model", [Generator(), Discriminator()])
@torch.inference_mode()
def test_different_batch_sizes(model: nn.Module) -> None:
    """
    Check if model outputs are the same for different batch sizes
    :param model: Model to test
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_sizes = [1, 4, 8]
    max_batch = max(batch_sizes)
    inputs = torch.testing.make_tensor((max_batch, 3, 256, 256), dtype=torch.float, device=device)
    fill_value = random.randint(0, 255)  # To keep all inputs the same
    outputs = []

    model = model.to(device)
    for batch_size in batch_sizes:
        batch_inputs = torch.split(inputs, batch_size, dim=0)
        batch_outputs = [model(batch_input) for batch_input in batch_inputs]
        outputs.append(torch.cat(batch_outputs, dim=0))

    for tensor_1, tensor_2 in itertools.pairwise(outputs):
        torch.testing.assert_close(tensor_1, tensor_2)
