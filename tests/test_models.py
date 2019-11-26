import torch
from src.models import LinearModel


def test_linear_model():
    m = LinearModel(5, 3)
    x = torch.randn(4, 5)
    o = m(x)
    expected_shapes = [4, 3]
    actual_shapes = o.shape
    assert all(e == a for e, a in zip(expected_shapes, actual_shapes))
