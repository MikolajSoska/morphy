import torch

from cyclegan.model.discriminator import Discriminator


@torch.inference_mode()
def test_forward_pass() -> None:
    model = Discriminator()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    batch_size = 8
    inputs = torch.testing.make_tensor((batch_size, 3, 256, 256), dtype=torch.float, device=device)

    outputs = model(inputs)
    assert outputs.shape == (batch_size, 1, 30, 30)
