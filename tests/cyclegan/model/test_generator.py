import torch

from cyclegan.model.generator import Generator


@torch.inference_mode()
def test_forward_pass() -> None:
    model = Generator()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    inputs = torch.testing.make_tensor((8, 3, 256, 256), dtype=torch.float, device=device)
    outputs = model(inputs)
    assert inputs.shape == outputs.shape
