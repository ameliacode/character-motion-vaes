import pybullet as p
import gym
from gym.spaces import Discrete
import numpy as np
import functools
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers

try:
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from mocap_envs import *
except:
    from .mocap_envs import *

## Environment for single agent independent Learning in shared environment state


class TwoPlayerFightingEnv(TargetEnv):
    def __init__(self,
                 device,
                 pose_vae_path):
        self.num_parallel = 2  # number of agents
        self.player_index = 0
        self.opponent_index = 1

        super().__init__(self,
                           self.num_parallel,
                           device,
                           pose_vae_path)

    #
    # def calc_env_state(self, next_frame):
    #     pass
    #
    # def calc_progress_reward(self):
    #     pass
    #
    # def get_observation_components(self):
    #     pass