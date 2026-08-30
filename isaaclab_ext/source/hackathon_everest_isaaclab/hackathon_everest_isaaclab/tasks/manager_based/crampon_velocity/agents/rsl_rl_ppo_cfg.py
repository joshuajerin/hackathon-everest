from dataclasses import field

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (
    G1RoughPPORunnerCfg,
)


@configclass
class EverestRoughPPORunnerCfg(G1RoughPPORunnerCfg):
    experiment_name = "everest_g1_crampon_rough"


@configclass
class EverestStatefulPPORunnerCfg(G1RoughPPORunnerCfg):
    experiment_name = "everest_g1_crampon_stateful"


@configclass
class EverestStandPPORunnerCfg(G1RoughPPORunnerCfg):
    experiment_name = "everest_g1_crampon_stand"
    save_interval = 10


@configclass
class EverestWalkPPORunnerCfg(G1RoughPPORunnerCfg):
    experiment_name = "everest_g1_crampon_walk"
    save_interval = 10


@configclass
class EverestSuitePPORunnerCfg(G1RoughPPORunnerCfg):
    experiment_name = "everest_g1_crampon_suite"
    save_interval = 10


@configclass
class EverestStandRandomizedPPORunnerCfg(EverestStandPPORunnerCfg):
    experiment_name = "everest_g1_crampon_stand_randomized"


@configclass
class EverestWalkRandomizedPPORunnerCfg(EverestWalkPPORunnerCfg):
    experiment_name = "everest_g1_crampon_walk_randomized"


@configclass
class EverestFrontPointRandomizedPPORunnerCfg(EverestWalkRandomizedPPORunnerCfg):
    experiment_name = "everest_g1_crampon_front_point_randomized"


@configclass
class EverestFrontPointStandRandomizedPPORunnerCfg(EverestStandRandomizedPPORunnerCfg):
    experiment_name = "everest_g1_crampon_front_point_stand_randomized"


@configclass
class EverestBoundedResidualActorCfg(RslRlMLPModelCfg):
    class_name: str = "hackathon_everest_isaaclab.learning.residual_policy:BoundedResidualMLPModel"
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    activation: str = "elu"
    obs_normalization: bool = False
    distribution_cfg: RslRlMLPModelCfg.GaussianDistributionCfg = (
        RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.02, std_type="scalar")
    )
    stock_checkpoint_path: str = ""
    stock_checkpoint_sha256: str = (
        "e0834ec91a204855ea681fa50cc19ebca59799ccca06cc104b0d2aee55068f49"
    )
    maximum_residual: float = 0.12
    expected_obs_dim: int = 310
    expected_action_dim: int = 37


@configclass
class EverestFrontPointBoundedResidualPPORunnerCfg(EverestWalkRandomizedPPORunnerCfg):
    experiment_name = "everest_g1_crampon_front_point_bounded_residual"
    save_interval = 10
    actor: EverestBoundedResidualActorCfg = EverestBoundedResidualActorCfg()

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.entropy_coef = 0.0
