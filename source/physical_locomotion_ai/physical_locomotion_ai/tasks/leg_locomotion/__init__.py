"""Locomotion environments with velocity-tracking commands.

These environments are based on the `legged_gym` environments provided by Rudin et al.

Reference:
    https://github.com/leggedrobotics/legged_gym
"""

import gymnasium as gym

from . import agents
##
# Register Gym environments.
##

gym.register(
    id='oneleg-goal',
    entry_point=f"{__name__}.one_leg_env:OneLegReachGoalTask",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.one_leg_env_cfg:OneLegEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OneLegFlatPPORunnerCfg",
    },
)

gym.register(
    id='oneleg-position',
    entry_point=f"{__name__}.one_leg_p2p_env:OneLegP2PTask",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.one_leg_env_cfg:OneLegP2PEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OneLegFlatPPORunnerCfg",
    },
)

gym.register(
    id="oneleg-uan",
    entry_point=f"{__name__}.one_leg_uan_env:OneLegUANCalibrationTask",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.one_leg_uan_env_cfg:OneLegUANEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OneLegUANPPORunnerCfg",
    },
)

gym.register(
    id="oneleg-uan-deploy",
    entry_point=f"{__name__}.one_leg_uan_deploy_env:OneLegUANDeployableCalibrationTask",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.one_leg_uan_deploy_env_cfg:OneLegUANDeployableEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OneLegUANDeployablePPORunnerCfg",
    },
)
