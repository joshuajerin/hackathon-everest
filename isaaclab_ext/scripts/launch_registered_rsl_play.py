#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
import sys

from hackathon_everest_isaaclab.runtime import acquire_isaac_process_lock
from hackathon_everest_isaaclab.tasks import register_cli
from rsl_rl.runners import OnPolicyRunner

_ISAAC_PROCESS_LOCK = acquire_isaac_process_lock()
register_cli()
if os.environ.get("EVEREST_EXPORT_ONLY") == "1":
    original_get_inference_policy = OnPolicyRunner.get_inference_policy

    def export_only_policy(self, *args, **kwargs):
        original_get_inference_policy(self, *args, **kwargs)

        def stop_after_export(_observation):
            self.env.close()
            raise KeyboardInterrupt

        return stop_after_export

    OnPolicyRunner.get_inference_policy = export_only_policy

play_directory = "/home/ubuntu/everest/IsaacLab3/scripts/reinforcement_learning/rsl_rl"
sys.path.insert(0, play_directory)
runpy.run_path(os.path.join(play_directory, "play.py"), run_name="__main__")
