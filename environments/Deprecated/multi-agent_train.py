import pybullet as p
import sys
from pathlib import Path
from os import path
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy

try:
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from mocap_renderer import *
    from multi_agent_pettingzoo import *
except:
    from .mocap_renderer import *
    from .multi_agent_pettingzoo import *


def train(model_path, timesteps=1e7):
    env = AdversePlayersFightingEnv(device=0, pose_vae_path=model_path)
    obs_list = env.reset()
    action_list = []
    for epoch in range(timesteps):
        for agent_index in range(env.num_parallel):
            obs = obs_list[agent_index]



def main(mode):
    current_dir = Path(__file__).resolve().parents[0]
    model_path = current_dir / "model"
    if mode == "train":
        train(model_path=model_path)

if __name__ == "__main__":
    main(mode="train")