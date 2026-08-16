# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg


def _legacy_on_policy_cfg(obs_groups=MISSING):
    kwargs = {
        "num_steps_per_env": 24,
        "max_iterations": 1,
        "save_interval": 1,
        "experiment_name": "test",
        "policy": RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=[32],
            critic_hidden_dims=[32],
            activation="elu",
        ),
    }
    if obs_groups is not MISSING:
        kwargs["obs_groups"] = obs_groups
    return RslRlOnPolicyRunnerCfg(**kwargs)


def test_missing_obs_groups_becomes_empty_mapping_for_rsl_rl_v4():
    cfg = _legacy_on_policy_cfg()

    handle_deprecated_rsl_rl_cfg(cfg, "4.0.0")

    assert cfg.obs_groups == {}
    assert cfg.to_dict()["obs_groups"] == {}


def test_explicit_obs_groups_are_preserved_for_rsl_rl_v4():
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    cfg = _legacy_on_policy_cfg(obs_groups=obs_groups)

    handle_deprecated_rsl_rl_cfg(cfg, "4.0.0")

    assert cfg.obs_groups == obs_groups
