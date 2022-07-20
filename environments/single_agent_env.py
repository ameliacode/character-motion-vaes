import pybullet as p
import gym
from gym.spaces import Discrete
from pathlib import Path
import numpy as np
import torch
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


