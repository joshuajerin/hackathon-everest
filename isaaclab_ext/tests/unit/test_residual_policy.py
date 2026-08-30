from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")
TensorDict = pytest.importorskip("tensordict").TensorDict

SOURCE = Path(__file__).parents[2] / "source/hackathon_everest_isaaclab"
sys.path.insert(0, str(SOURCE))

from hackathon_everest_isaaclab.learning.residual_policy import BoundedResidualMLPModel
from rsl_rl.modules.mlp import MLP


def _checkpoint(tmp_path: Path) -> tuple[Path, MLP, str]:
    stock = MLP(310, 37, (512, 256, 128), "elu")
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {"actor_state_dict": {f"mlp.{key}": value for key, value in stock.state_dict().items()}},
        path,
    )
    return path, stock, hashlib.sha256(path.read_bytes()).hexdigest()


def _model(tmp_path: Path) -> tuple[BoundedResidualMLPModel, MLP, TensorDict]:
    path, stock, sha256 = _checkpoint(tmp_path)
    obs = TensorDict({"policy": torch.randn(5, 310)}, batch_size=[5])
    model = BoundedResidualMLPModel(
        obs=obs,
        obs_groups={"actor": ["policy"]},
        obs_set="actor",
        output_dim=37,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 0.01,
            "std_type": "scalar",
        },
        stock_checkpoint_path=str(path),
        stock_checkpoint_sha256=sha256,
        maximum_residual=0.12,
    )
    return model, stock, obs


def test_zero_initialized_residual_is_exact_stock_and_exports(tmp_path: Path) -> None:
    model, stock, obs = _model(tmp_path)
    expected = stock(obs["policy"])
    actual = model(obs)
    assert torch.equal(actual, expected)
    scripted = torch.jit.script(model.as_jit())
    assert torch.equal(scripted(obs["policy"]), expected)
    assert all(not parameter.requires_grad for parameter in model.stock_mlp.parameters())


def test_residual_is_bounded_and_stock_stays_frozen(tmp_path: Path) -> None:
    model, _, obs = _model(tmp_path)
    with torch.no_grad():
        linear_layers = [module for module in model.mlp if isinstance(module, torch.nn.Linear)]
        linear_layers[-1].bias.fill_(20.0)
    stock_action = model.stock_mlp(obs["policy"])
    action = model(obs)
    assert bool(((action - stock_action).abs() <= 0.120001).all())
    action.sum().backward()
    assert all(parameter.grad is None for parameter in model.stock_mlp.parameters())
    assert any(parameter.grad is not None for parameter in model.mlp.parameters())


def test_model_rejects_wrong_hash_and_dimensions(tmp_path: Path) -> None:
    path, _, sha256 = _checkpoint(tmp_path)
    obs = TensorDict({"policy": torch.randn(2, 310)}, batch_size=[2])
    common = {
        "obs": obs,
        "obs_groups": {"actor": ["policy"]},
        "obs_set": "actor",
        "output_dim": 37,
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 0.01,
            "std_type": "scalar",
        },
        "stock_checkpoint_path": str(path),
    }
    with pytest.raises(ValueError, match="hash mismatch"):
        BoundedResidualMLPModel(**common, stock_checkpoint_sha256="0" * 64)
    bad_obs = TensorDict({"policy": torch.randn(2, 309)}, batch_size=[2])
    with pytest.raises(ValueError, match="observation dimension"):
        BoundedResidualMLPModel(
            **{**common, "obs": bad_obs, "distribution_cfg": dict(common["distribution_cfg"])},
            stock_checkpoint_sha256=sha256,
        )
