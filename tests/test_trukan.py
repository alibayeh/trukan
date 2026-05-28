import pytest
import torch
from src.trukan.trukan_layer import TruKanLayer


def test_output_shape():
    module = TruKanLayer(in_dim=10, out_dim=5)

    x_batch = torch.randn(32, 10)
    out_batch = module(x_batch)
    assert out_batch.shape == (32, 5)


def test_output_range_and_no_nan():
    module = TruKanLayer(in_dim=10, out_dim=5)
    x = torch.randn(32, 10)
    out = module(x)

    assert not torch.any(torch.isnan(out))
    assert not torch.any(torch.isinf(out))

    # torch.testing.assert_close(out, torch.clamp(module.linear(x), 0, 1))


def test_parameters_update_after_backward():
    module = TruKanLayer(in_dim=10, out_dim=5)
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
    params_before = {name: param.clone() for name, param in module.named_parameters()}
    x = torch.randn(100, 10)
    loss = module(x).sum()
    loss.backward()
    optimizer.step()

    for name, param in module.named_parameters():
        assert not torch.allclose(param, params_before[name]), f"{name} did not change!"


def test_device_compatibility():
    if torch.cuda.is_available():
        module = TruKanLayer(in_dim=10, out_dim=5).cuda()
        x = torch.randn(8, 10).cuda()
        out = module(x)
        assert out.device.type == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
