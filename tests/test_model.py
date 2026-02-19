
import torch
from src.model import SimpleCNN

def test_model_forward_pass():
    model = SimpleCNN()
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)

    assert output.shape == (1, 2)
