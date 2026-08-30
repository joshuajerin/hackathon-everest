import gymnasium as gym

_TASK_PREFIX = "Everest-Velocity-Rough-G1-Crampon"

if f"{_TASK_PREFIX}-v0" not in gym.registry:
    gym.register(
        id=f"{_TASK_PREFIX}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:EverestG1CramponRoughEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:EverestRoughPPORunnerCfg",
        },
    )
    gym.register(
        id=f"{_TASK_PREFIX}-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:EverestG1CramponRoughEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:EverestRoughPPORunnerCfg",
        },
    )

_STATEFUL_PREFIX = "Everest-Velocity-Flat-G1-Crampon-Stateful"

if f"{_STATEFUL_PREFIX}-v0" not in gym.registry:
    gym.register(
        id=f"{_STATEFUL_PREFIX}-v0",
        entry_point=f"{__name__}.stateful_env:EverestStatefulCramponEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:EverestG1CramponStatefulEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:EverestStatefulPPORunnerCfg",
        },
    )
    gym.register(
        id=f"{_STATEFUL_PREFIX}-Play-v0",
        entry_point=f"{__name__}.stateful_env:EverestStatefulCramponEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:EverestG1CramponStatefulEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:EverestStatefulPPORunnerCfg",
        },
    )

_BOOTSTRAP_STAND_TASK = "Everest-Velocity-Flat-G1-Crampon-Stateful-Bootstrap-Stand-v0"
if _BOOTSTRAP_STAND_TASK not in gym.registry:
    gym.register(
        id=_BOOTSTRAP_STAND_TASK,
        entry_point=f"{__name__}.stateful_env:EverestStatefulCramponEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:EverestG1CramponStatefulBootstrapStandEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:EverestStandPPORunnerCfg",
        },
    )

_BOOTSTRAP_TASK = "Everest-Velocity-Flat-G1-Crampon-Stateful-Bootstrap-v0"
if _BOOTSTRAP_TASK not in gym.registry:
    gym.register(
        id=_BOOTSTRAP_TASK,
        entry_point=f"{__name__}.stateful_env:EverestStatefulCramponEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:EverestG1CramponStatefulBootstrapEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:EverestWalkPPORunnerCfg",
        },
    )

_SUITE_TASK = "Everest-Velocity-Suite-G1-Crampon-Stateful-v0"
if _SUITE_TASK not in gym.registry:
    gym.register(
        id=_SUITE_TASK,
        entry_point=f"{__name__}.stateful_env:EverestStatefulCramponEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:EverestG1CramponStatefulSuiteEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:EverestSuitePPORunnerCfg",
        },
    )


_BOOTSTRAP_STAND_RANDOMIZED_TASK = (
    "Everest-Velocity-Flat-G1-Crampon-Stateful-Bootstrap-Stand-Randomized-v0"
)
if _BOOTSTRAP_STAND_RANDOMIZED_TASK not in gym.registry:
    gym.register(
        id=_BOOTSTRAP_STAND_RANDOMIZED_TASK,
        entry_point=f"{__name__}.stateful_env:EverestStatefulCramponEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:EverestG1CramponStatefulBootstrapStandRandomizedEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:EverestStandRandomizedPPORunnerCfg",
        },
    )

_BOOTSTRAP_RANDOMIZED_TASK = "Everest-Velocity-Flat-G1-Crampon-Stateful-Bootstrap-Randomized-v0"
if _BOOTSTRAP_RANDOMIZED_TASK not in gym.registry:
    gym.register(
        id=_BOOTSTRAP_RANDOMIZED_TASK,
        entry_point=f"{__name__}.stateful_env:EverestStatefulCramponEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:EverestG1CramponStatefulBootstrapRandomizedEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:EverestWalkRandomizedPPORunnerCfg",
        },
    )


_BOOTSTRAP_FRONT_POINT_STAND_RANDOMIZED_TASK = (
    "Everest-Velocity-Flat-G1-Crampon-Stateful-Bootstrap-FrontPoint-Stand-Randomized-v0"
)
if _BOOTSTRAP_FRONT_POINT_STAND_RANDOMIZED_TASK not in gym.registry:
    gym.register(
        id=_BOOTSTRAP_FRONT_POINT_STAND_RANDOMIZED_TASK,
        entry_point=f"{__name__}.stateful_env:EverestStatefulCramponEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.env_cfg:"
                "EverestG1CramponStatefulBootstrapFrontPointStandRandomizedEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                f"{__name__}.agents.rsl_rl_ppo_cfg:EverestFrontPointStandRandomizedPPORunnerCfg"
            ),
        },
    )


_BOOTSTRAP_FRONT_POINT_RANDOMIZED_TASK = (
    "Everest-Velocity-Flat-G1-Crampon-Stateful-Bootstrap-FrontPoint-Randomized-v0"
)
if _BOOTSTRAP_FRONT_POINT_RANDOMIZED_TASK not in gym.registry:
    gym.register(
        id=_BOOTSTRAP_FRONT_POINT_RANDOMIZED_TASK,
        entry_point=f"{__name__}.stateful_env:EverestStatefulCramponEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.env_cfg:EverestG1CramponStatefulBootstrapFrontPointRandomizedEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                f"{__name__}.agents.rsl_rl_ppo_cfg:EverestFrontPointRandomizedPPORunnerCfg"
            ),
        },
    )


_FRONT_POINT_BOUNDED_RESIDUAL_TASK = (
    "Everest-Velocity-Flat-G1-Crampon-Stateful-FrontPoint-BoundedResidual-v0"
)
if _FRONT_POINT_BOUNDED_RESIDUAL_TASK not in gym.registry:
    gym.register(
        id=_FRONT_POINT_BOUNDED_RESIDUAL_TASK,
        entry_point=f"{__name__}.stateful_env:EverestStatefulCramponEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.env_cfg:EverestG1CramponStatefulBootstrapFrontPointRandomizedEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                f"{__name__}.agents.rsl_rl_ppo_cfg:EverestFrontPointBoundedResidualPPORunnerCfg"
            ),
        },
    )


_FRONT_POINT_LOAD_CORRECTED_RESIDUAL_TASK = (
    "Everest-Velocity-Flat-G1-Crampon-Stateful-FrontPoint-LoadCorrectedResidual-v0"
)
if _FRONT_POINT_LOAD_CORRECTED_RESIDUAL_TASK not in gym.registry:
    gym.register(
        id=_FRONT_POINT_LOAD_CORRECTED_RESIDUAL_TASK,
        entry_point=f"{__name__}.stateful_env:EverestStatefulCramponEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.env_cfg:"
                "EverestG1CramponStatefulBootstrapFrontPointLoadCorrectedRandomizedEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                f"{__name__}.agents.rsl_rl_ppo_cfg:EverestFrontPointBoundedResidualPPORunnerCfg"
            ),
        },
    )


_FRONT_POINT_POSITIVE_TRACKING_RESIDUAL_TASK = (
    "Everest-Velocity-Flat-G1-Crampon-Stateful-FrontPoint-PositiveTrackingResidual-v0"
)
if _FRONT_POINT_POSITIVE_TRACKING_RESIDUAL_TASK not in gym.registry:
    gym.register(
        id=_FRONT_POINT_POSITIVE_TRACKING_RESIDUAL_TASK,
        entry_point=f"{__name__}.stateful_env:EverestStatefulCramponEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.env_cfg:"
                "EverestG1CramponStatefulBootstrapFrontPointPositiveTrackingRandomizedEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                f"{__name__}.agents.rsl_rl_ppo_cfg:EverestFrontPointBoundedResidualPPORunnerCfg"
            ),
        },
    )


_FRONT_POINT_LINEAR_TRACKING_RESIDUAL_TASK = (
    "Everest-Velocity-Flat-G1-Crampon-Stateful-FrontPoint-LinearTrackingResidual-v0"
)
if _FRONT_POINT_LINEAR_TRACKING_RESIDUAL_TASK not in gym.registry:
    gym.register(
        id=_FRONT_POINT_LINEAR_TRACKING_RESIDUAL_TASK,
        entry_point=f"{__name__}.stateful_env:EverestStatefulCramponEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.env_cfg:"
                "EverestG1CramponStatefulBootstrapFrontPointLinearTrackingRandomizedEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                f"{__name__}.agents.rsl_rl_ppo_cfg:EverestFrontPointBoundedResidualPPORunnerCfg"
            ),
        },
    )
