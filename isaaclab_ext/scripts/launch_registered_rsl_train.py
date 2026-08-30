#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import runpy
import sys

from hackathon_everest_isaaclab.runtime import acquire_isaac_process_lock
from hackathon_everest_isaaclab.tasks import register_cli
from rsl_rl.runners import OnPolicyRunner

_ISAAC_PROCESS_LOCK = acquire_isaac_process_lock()
_training_requested = "--max_iterations" in sys.argv
register_cli()
original_learn = OnPolicyRunner.learn
_training_status = {"completed": False}


def learn_with_visible_interrupt(self, *args, **kwargs):
    try:
        result = original_learn(self, *args, **kwargs)
        _training_status["completed"] = True
        return result
    except KeyboardInterrupt as error:
        raise RuntimeError(
            "Isaac/RSL training was interrupted before its requested iteration count"
        ) from error


OnPolicyRunner.learn = learn_with_visible_interrupt
if os.environ.get("EVEREST_ACTOR_ONLY_RESUME") == "1":
    original_load = OnPolicyRunner.load

    def load_policy_weights_only(self, path, load_cfg=None, strict=True, map_location=None):
        selected = {
            "actor": True,
            "critic": False,
            "optimizer": False,
            "iteration": False,
            "rnd": False,
        }
        result = original_load(
            self,
            path,
            load_cfg=selected,
            strict=strict,
            map_location=map_location,
        )
        initial_std = float(os.environ.get("EVEREST_INITIAL_ACTION_STD", "0.02"))
        adaptation_lr = float(os.environ.get("EVEREST_ADAPTATION_LR", "0.00003"))
        entropy_coef = float(os.environ.get("EVEREST_ENTROPY_COEF", "0.0"))
        if initial_std <= 0.0 or adaptation_lr <= 0.0 or entropy_coef < 0.0:
            raise ValueError("Invalid Everest actor-only adaptation hyperparameter")
        policy = self.alg.get_policy()
        distribution = policy.distribution
        import torch

        with torch.no_grad():
            if distribution.std_type == "scalar":
                distribution.std_param.fill_(initial_std)
            else:
                distribution.log_std_param.fill_(math.log(initial_std))
        self.alg.learning_rate = adaptation_lr
        self.alg.entropy_coef = entropy_coef
        for group in self.alg.optimizer.param_groups:
            group["lr"] = adaptation_lr
        critic_warmup = int(os.environ.get("EVEREST_CRITIC_WARMUP_ITERATIONS", "30"))
        trainable_linear_layers = int(os.environ.get("EVEREST_ACTOR_TRAINABLE_LINEAR_LAYERS", "1"))
        if critic_warmup < 0 or trainable_linear_layers < 0:
            raise ValueError("Invalid Everest warm-up or actor-layer count")

        def set_actor_trainable(enabled: bool) -> None:
            for parameter in policy.parameters():
                parameter.requires_grad_(enabled and trainable_linear_layers == 0)
            if enabled and trainable_linear_layers:
                from torch import nn

                linear_layers = [module for module in policy.mlp if isinstance(module, nn.Linear)]
                if trainable_linear_layers > len(linear_layers):
                    raise ValueError("Requested more trainable actor layers than available")
                for layer in linear_layers[-trainable_linear_layers:]:
                    for parameter in layer.parameters():
                        parameter.requires_grad_(True)

        if critic_warmup:
            set_actor_trainable(False)
            original_update = self.alg.update
            warmup_state = {"updates": 0}

            def update_with_critic_warmup(*args, **kwargs):
                result = original_update(*args, **kwargs)
                warmup_state["updates"] += 1
                if warmup_state["updates"] == critic_warmup:
                    set_actor_trainable(True)
                    print(
                        f"[Everest] Actor unfrozen after {critic_warmup} critic updates "
                        f"(trainable_linear_layers={trainable_linear_layers})",
                        flush=True,
                    )
                return result

            self.alg.update = update_with_critic_warmup
        else:
            set_actor_trainable(True)
        print(
            "[Everest] Actor-only adaptation: "
            f"initial_std={initial_std} learning_rate={adaptation_lr} "
            f"entropy_coef={entropy_coef} critic_warmup={critic_warmup} "
            f"trainable_linear_layers={trainable_linear_layers}",
            flush=True,
        )
        return result

    OnPolicyRunner.load = load_policy_weights_only
elif os.environ.get("EVEREST_RESUME_LEARNING_RATE"):
    original_load = OnPolicyRunner.load

    def load_with_learning_rate_override(self, path, load_cfg=None, strict=True, map_location=None):
        result = original_load(
            self,
            path,
            load_cfg=load_cfg,
            strict=strict,
            map_location=map_location,
        )
        learning_rate = float(os.environ["EVEREST_RESUME_LEARNING_RATE"])
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("EVEREST_RESUME_LEARNING_RATE must be finite and positive")
        self.alg.learning_rate = learning_rate
        for group in self.alg.optimizer.param_groups:
            group["lr"] = learning_rate
        print(
            f"[Everest] Resumed optimizer learning rate overridden to {learning_rate}",
            flush=True,
        )
        return result

    OnPolicyRunner.load = load_with_learning_rate_override
sys.path.insert(0, "/home/ubuntu/everest/IsaacLab3/scripts/reinforcement_learning/rsl_rl")
try:
    runpy.run_path(
        "/home/ubuntu/everest/IsaacLab3/scripts/reinforcement_learning/rsl_rl/train.py",
        run_name="__main__",
    )
except SystemExit as error:
    if _training_requested and not _training_status["completed"]:
        raise RuntimeError(
            "Isaac training exited before completing the requested iterations"
        ) from error
    raise
if _training_requested and not _training_status["completed"]:
    raise RuntimeError("Isaac training returned before completing the requested iterations")
