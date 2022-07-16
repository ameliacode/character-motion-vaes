import pybullet as p
import gym

try:
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from mocap_envs import *
except:
    from .mocap_envs import *

## Environment for single agent independent Learning in shared environment state


class SingleTargetEnv(TargetEnv):
    def __init__(self,
                 num_parallel,
                 device,
                 pose_vae_path):
        TargetEnv.__init__(self,
                           num_parallel,
                           device,
                           pose_vae_path)

    def calc_env_state(self, next_frame):
        pass