try:
    from .mocap_envs import *
    from ..common.bullet_objects import *
    from ..common.bullet_utils import *
    from .mocap_renderer import *
    from .single_agent_env import *
except:
    import sys
    from os import path
    current_dir = path.dirname(path.abspath(__file__))
    parent_dir = path.dirname(current_dir)
    sys.path.append(parent_dir)
    from environments.mocap_envs import *
    from environments.mocap_renderer import *
    from environments.single_agent_env import *
    from common.bullet_objects import *
    from common.bullet_utils import *

def test_env():
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    num_parallel = 3
    device = "cpu"
    current_dir = Path(__file__).resolve().parents[1]
    mvae_dir = current_dir / "vae_motion" / "models" / "posevae_c1_e6_l32.pt"
    env = PunchingPlayerEnv(num_parallel=num_parallel,
                            device=device,
                            pose_vae_path=str(mvae_dir))
    obs = env.reset()
    ep_reward = 0
    while True:
        # controller = torch.load(controller_dir, map_location=device)
        for _ in range(1000):
            # action = controller.predict(obs)
            action = env.action_space.sample()
            action = torch.tensor([action] * num_parallel).to(device)
            # print(action)
            env.step(action)
            time.sleep(0.01)

def test_controller():
    pass

def main():
    test_env()

if __name__ == "__main__":
    main()