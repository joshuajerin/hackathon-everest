from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules.mlp import MLP
from rsl_rl.utils import unpad_trajectories
from torch import nn

OFFICIAL_G1_ROUGH_CHECKPOINT_SHA256 = (
    "e0834ec91a204855ea681fa50cc19ebca59799ccca06cc104b0d2aee55068f49"
)


class _BoundedResidualJit(nn.Module):
    def __init__(self, model: BoundedResidualMLPModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.stock_mlp = copy.deepcopy(model.stock_mlp)
        self.residual_mlp = copy.deepcopy(model.mlp)
        self.register_buffer("residual_limit", model.residual_limit.detach().clone())

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        latent = self.obs_normalizer(observation)
        stock_action = self.stock_mlp(latent)
        residual = self.residual_limit * torch.tanh(self.residual_mlp(latent))
        return stock_action + residual

    @torch.jit.export
    def reset(self) -> None:
        pass


class _BoundedResidualOnnx(_BoundedResidualJit):
    is_recurrent: bool = False

    def __init__(self, model: BoundedResidualMLPModel, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose
        self.input_size = model.obs_dim

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]


class BoundedResidualMLPModel(MLPModel):
    """Frozen official stock actor plus a zero-initialized bounded residual policy."""

    def __init__(
        self,
        obs,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        stock_checkpoint_path: str = "",
        stock_checkpoint_sha256: str = OFFICIAL_G1_ROUGH_CHECKPOINT_SHA256,
        maximum_residual: float = 0.12,
        expected_obs_dim: int = 310,
        expected_action_dim: int = 37,
    ) -> None:
        if obs_normalization:
            raise ValueError("bounded residual actor requires raw stock observations")
        if output_dim != expected_action_dim:
            raise ValueError(
                f"bounded residual action dimension must be {expected_action_dim}, got {output_dim}"
            )
        if not 0.0 < maximum_residual <= 0.35:
            raise ValueError("maximum_residual must be in (0, 0.35]")
        distribution_cfg = copy.deepcopy(distribution_cfg)
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )
        if self.obs_dim != expected_obs_dim:
            raise ValueError(
                f"bounded residual observation dimension must be {expected_obs_dim}, got {self.obs_dim}"
            )
        if self.distribution is None or type(self.distribution).__name__ != "GaussianDistribution":
            raise ValueError("bounded residual actor requires GaussianDistribution")
        configured_path = stock_checkpoint_path or os.environ.get(
            "EVEREST_STOCK_RSL_CHECKPOINT", ""
        )
        if not configured_path:
            raise ValueError(
                "Set stock_checkpoint_path or EVEREST_STOCK_RSL_CHECKPOINT for residual training"
            )
        checkpoint_path = Path(configured_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Stock RSL checkpoint not found: {checkpoint_path}")
        actual_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if actual_hash != stock_checkpoint_sha256:
            raise ValueError(
                f"Stock checkpoint hash mismatch: expected {stock_checkpoint_sha256}, got {actual_hash}"
            )

        self.register_buffer(
            "residual_limit",
            torch.full((output_dim,), float(maximum_residual), dtype=torch.float32),
        )
        self.stock_mlp = MLP(self.obs_dim, output_dim, (512, 256, 128), "elu")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        actor_state = checkpoint.get("actor_state_dict")
        if not isinstance(actor_state, dict):
            raise TypeError("Stock checkpoint is missing actor_state_dict")
        stock_state = {
            key.removeprefix("mlp."): value
            for key, value in actor_state.items()
            if key.startswith("mlp.")
        }
        expected_stock_keys = set(self.stock_mlp.state_dict())
        if set(stock_state) != expected_stock_keys:
            raise ValueError("Stock actor MLP keys do not match the pinned 310-to-37 architecture")
        self.stock_mlp.load_state_dict(stock_state, strict=True)
        self.stock_mlp.requires_grad_(False)
        self.stock_mlp.eval()

        residual_linear_layers = [module for module in self.mlp if isinstance(module, nn.Linear)]
        if not residual_linear_layers:
            raise RuntimeError("Residual MLP has no linear output layer")
        nn.init.zeros_(residual_linear_layers[-1].weight)
        nn.init.zeros_(residual_linear_layers[-1].bias)

    def train(self, mode: bool = True):
        result = super().train(mode)
        self.stock_mlp.eval()
        return result

    def forward(
        self,
        obs,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        latent = self.get_latent(obs, masks=None, hidden_state=hidden_state)
        stock_action = self.stock_mlp(latent).detach()
        residual = self.residual_limit * torch.tanh(self.mlp(latent))
        mean_action = stock_action + residual
        if stochastic_output:
            self.distribution.update(mean_action)
            return self.distribution.sample()
        return self.distribution.deterministic_output(mean_action)

    def as_jit(self) -> nn.Module:
        return _BoundedResidualJit(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _BoundedResidualOnnx(self, verbose)
