import os

current_dir = os.path.dirname(os.path.realpath(__file__))

import gym
from gym.envs.registration import registry, make, spec


def register(id, *args, **kvargs):
    if id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, *args, **kvargs)


register(id="RandomWalkEnv-v0", entry_point="environments.mocap_envs:RandomWalkEnv")
register(id="TargetEnv-v0", entry_point="environments.mocap_envs:TargetEnv")
register(id="JoystickEnv-v0", entry_point="environments.mocap_envs:JoystickEnv")
register(id="PathFollowEnv-v0", entry_point="environments.mocap_envs:PathFollowEnv")
register(id="HumanMazeEnv-v0", entry_point="environments.mocap_envs:HumanMazeEnv")
register(id="CustomPunchingEnv-v0", entry_point="environments.custom_env:CustomPunchingEnv")
register(id="CustomTargetEnv-v0", entry_point="environments.custom_env:CustomTargetEnv")
register(id="AdversePlayersFightingEnv-v0", entry_point="environments.DEPRECATED.multi_agent_env:AdversePlayersFightingEnv")
