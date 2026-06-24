import pytest
import torch
from src.trukan.trukan_layer import TruKanLayer


@pytest.fixture(
    params=[
        {"name": "fixed_shared3", "num_knots": 3, "learn": False, "shared": True},
        {"name": "fixed_shared5", "num_knots": 5, "learn": False, "shared": True},
        {"name": "fixed_individual3", "num_knots": 3, "learn": False, "shared": False},
        {"name": "fixed_individual5", "num_knots": 5, "learn": False, "shared": False},
        {"name": "learn_shared3", "num_knots": 3, "learn": True, "shared": True},
        {"name": "learn_shared5", "num_knots": 5, "learn": True, "shared": True},
        {"name": "learn_individual3", "num_knots": 3, "learn": True, "shared": False},
        {"name": "learn_individual5", "num_knots": 5, "learn": True, "shared": False},
    ]
)
def module(request):
    config = request.param
    learn = config["learn"]
    shared = config["shared"]
    module = TruKanLayer(
        in_dim=10,
        out_dim=5,
        learn_knots=learn,
        shared_knots=shared,
    )
    yield module


def test_output_shape(module):
    x_batch = torch.randn(32, 10)
    out_batch = module(x_batch)
    assert out_batch.shape == (32, 5)


def test_output_range_and_no_nan(module):
    x = torch.randn(32, 10)
    out = module(x)

    assert not torch.any(torch.isnan(out))
    assert not torch.any(torch.isinf(out))

    # torch.testing.assert_close(out, torch.clamp(module.linear(x), 0, 1))


def test_parameters_update_after_backward(module):
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
    params_before = {name: param.clone() for name, param in module.named_parameters()}
    x = torch.randn(100, 10)
    loss = module(x).sum()
    loss.backward()
    optimizer.step()

    for name, param in module.named_parameters():
        assert not torch.allclose(param, params_before[name]), f"{name} did not change!"


def test_device_compatibility(module):
    if torch.cuda.is_available():
        module = module.cuda()
        x = torch.randn(8, 10).cuda()
        out = module(x)
        assert out.device.type == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
