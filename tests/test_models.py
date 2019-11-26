import torch
from src.models import LinearModel, MultiInputLstm


def test_linear_model():
    m = LinearModel(5, 3)
    x = torch.randn(4, 5)
    o = m(x)
    expected_shapes = [4, 3]
    actual_shapes = o.shape
    assert all(e == a for e, a in zip(expected_shapes, actual_shapes))


def test_multi_input_lstm():
    m = MultiInputLstm(10, 16, 9)
    t = torch.randint(10, (3, 5))
    b = torch.randint(10, (3, 7))
    a = torch.randint(10, (3, 2))
    o = m(t, b, a)
    expected_shapes = [3, 9]
    actual_shapes = o.shape
    assert all(e == a for e, a in zip(expected_shapes, actual_shapes))
