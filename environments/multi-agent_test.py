import pybullet as p
from pathlib import Path
import torch
import time

try:
    from .mocap_envs import *
    from ..common.bullet_objects import *
    from ..common.bullet_utils import *
    from .mocap_renderer import *
    from .multi_agent_env import *
except:
    import sys
    from os import path
    current_dir = path.dirname(path.abspath(__file__))
    parent_dir = path.dirname(current_dir)
    sys.path.append(parent_dir)
    from environments.mocap_envs import *
    from environments.mocap_renderer import *
    from environments.multi_agent_pettingzoo import *
    from common.bullet_objects import *
    from common.bullet_utils import *

def test_env(mvae_dir, controller_dir):
    env = AdversePlayersFightingEnv()
    env.reset()
    while True:
        action = env.action_space.sample()
        env.step(action)
        env.render()
        time.sleep(0.01)

def test_controller(mvae_dir, controller_dir):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = AdversePlayersFightingEnv()
    obs = env.reset()
    ep_reward = 0
    while True:
        controller = torch.load(controller_dir, map_location=device)
        for _ in range(1000):
            action = controller.predict(obs)
            obs, rewards, done, info = env.step(action)
            ep_reward += rewards
            if done.any():
                ep_reward *= (~done).float()
                reset_indices = env.parallel_ind_buf.masked_select(done.squeeze())
                obs = env.reset(reset_indices)
            if info.get("reset"):
                print("--- Episode reward: %2.4f" % float(ep_reward.mean()))
                ep_reward = 0
                obs = env.reset()
            time.sleep(0.01)

def test_pettingzoo():
    env = AdversePlayersFightingEnv()
    env.reset()
    policy = lambda obs, agent: env._action_spaces[agent].sample()
    for agent in env.agent_iter(max_iter = 2000):
        observation, reward, done, info = env.last()
        action = policy(observation, agent)
        env.step(action)


def test_render(mvae_dir, controller_dir):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = AdversePlayersFightingEnv()
    # env = TestEnv()
    env.reset()
    while True:
        env.render()


def test_model(device, mvae_dir, timesteps=1e7): # execution of petting zoo implementation
    env = AdversePlayersFightingEnv(device=device, pose_vae_path=mvae_dir)
    policy = 0
    env.reset()
    for agent in env.agent_iter():
        observation, reward, done, info = env.last()
        action = policy(observation) if not done else None
        env.step(action)
    env.close()

def main():
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    current_dir = Path(__file__).resolve().parents[1]
    device="cpu"
    mvae_dir = current_dir / "vae_motion" / "models" / "posevae_c1_e6_l32.pt"
    controller_dir = current_dir / "vae_motion" / "con_TargetEnv-v0.pt"

    # test_env(mvae_dir=mvae_dir, controller_dir=controller_dir)
    # test_render(mvae_dir=mvae_dir, controller_dir=controller_dir)
    test_pettingzoo()

if __name__=="__main__":
    main()