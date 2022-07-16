import pybullet as p
from pathlib import Path
import torch
import time

try:
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from mocap_envs import *
    from multi_agent_target_env import *
except:
    from .mocap_envs import *
    from .multi_agent_target_env import *

def test_env(device, mvae_dir, controller_dir):
    env = TwoPlayerFightingEnv(device=device, pose_vae_path=mvae_dir)
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

def test_model(device, mvae_dir, timesteps=1e7): # execution of petting zoo implementation
    env = TwoPlayerFightingEnv(device=device, pose_vae_path=mvae_dir)
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

    test_env(device=device, mvae_dir=mvae_dir, controller_dir=controller_dir)


if __name__=="__main__":
    main()