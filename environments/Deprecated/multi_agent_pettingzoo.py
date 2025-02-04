import pybullet as p
import gym
from gym.spaces import Box
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
    current_dir = path.dirname(path.abspath(__file__))
    parent_dir = path.dirname(current_dir)
    sys.path.append(parent_dir)
    from environments.mocap_envs import *
    from environments.mocap_renderer import *
    from common.bullet_objects import *
    from common.bullet_utils import *
except:
    from .mocap_envs import *
    from ..common.bullet_objects import *
    from ..common.bullet_utils import *
    from .mocap_renderer import *

## Environment for single agent independent Learning in shared environment state


def env():
    pass

def raw_env():
    pass

class AdversePlayersFightingEnv(AECEnv, EnvBase): # raw env
    def __init__(self,
                 rendered=True,
                 use_params=False,
                 camera_tracking=True,
                 frame_skip=1):
        # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        self.num_parallel = 2 # number of characters in this case
        self.num_characters = 2

        self.current_dir = Path(__file__).resolve().parents[1]
        self.pose_vae_path = str(self.current_dir / "vae_motion" / "models" / "posevae_c1_e6_l32.pt")
        EnvBase.__init__(self, self.num_parallel,
                        self.device,
                        self.pose_vae_path,
                        rendered,
                        use_params,
                        camera_tracking,
                        frame_skip)
        self.arena_length = (-3,3)
        self.arena_width = (-3,3)

        self.glove_indices = [30,23] #right and left hand
        self.foot_indices = [11,6] #right and left toe
        self.target_indices = [16,13,14] #Head,Torso
        #temporary state container for velocity calculation
        self.glove_state_dim = 6 * 6 # len(target_indices) * len(foot_indices) * left+right * 3D coord

        # PettingZoo
        self.possible_agents=[0,1] #players
        self.agents = self.possible_agents[:]

        high = np.inf * np.ones([self.action_dim])
        self._action_spaces = {agent: Box(-high, high, dtype=np.float32) for agent in self.possible_agents}
        self.observation_dim = (
                self.frame_dim * self.num_condition_frames + self.action_dim
        )
        high = np.inf * np.ones([self.observation_dim])
        self._observation_spaces = {agent: Box(-high, high, dtype=np.float32) for agent in self.possible_agents}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent) -> Box:
        self.observation_dim = (
            self.frame_dim * self.num_condition_frames + self.action_dim
        )
        high = np.inf * np.ones([self.observation_dim])
        return Box(-high, high, dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent) -> Box:
        high = np.inf * np.ones([self.action_dim])
        return Box(-high, high, dtype=np.float32)

    def calc_glove_state(self, agent):  # player's glove and opponent target
        player = agent
        opponent = (agent + 1) % 2

        player_glove = [self.viewer.characters.ids[player*31 + self.glove_indices[idx]] for idx in range(len(self.glove_indices))]
        opponent_target = [self.viewer.characters.ids[opponent*31 + self.target_indices[idx]] for idx in range(len(self.target_indices))]

        agent_glove_idx = 0
        agent_glove_state = []
        for target_id in opponent_target: # Head, pelvis, abdomen respectively
            target_state = self.viewer._p.getBasePositionAndOrientation(target_id)
            for glove_id in player_glove: # right glove, left glove respectively
                glove_state = self.viewer._p.getBasePositionAndOrientation(glove_id)
                relative_pos = [target_state[0][idx] - glove_state[0][idx] for idx in range(3)]
                agent_glove_state.append(relative_pos)
                agent_glove_state.append([relative_pos[idx] - self.agents_glove_state[agent][agent_glove_idx][idx] for idx in range(3)])
                agent_glove_idx += 2

        self.agents_glove_state[agent] = agent_glove_state
        return agent_glove_state

    def get_root_delta_and_angle(self, agent): # player and opponent
        player = agent
        opponent = (agent + 1) % 2
        root_delta = self.root_xz[opponent] - self.root_xz[player]
        root_angle = (
            torch.atan2(root_delta[1], root_delta[0])
            + self.root_facing[player]
        )
        return root_delta, root_angle

    def get_attack_state(self, agent):  # player to opponent
        player = agent
        glove_state = self.agents_glove_state[player]
        return glove_state

    def get_defense_state(self, agent):  # opponent to player
        opponent = agent
        glove_state = self.agents_glove_state[opponent]
        return glove_state

    def observe(self, agent): # get observable components
        player = agent
        opponent = (agent + 1) % 2

        # root_delta, _ = self.get_root_delta_and_angle(player)
        # mat = self.get_rotation_matrix(-self.root_facing[player])
        # delta = (mat * root_delta.unsqueeze(1)).sum(dim=2)
        delta = 0.0
        condition = self.get_vae_condition(normalize=False)

        attack_glove = self.get_attack_state(player)
        defense_glove = self.get_defense_state(opponent)

        # return condition, delta, attack_foot, attack_glove, defense_foot, defense_glove
        # return condition, delta, attack_glove, defense_glove
        return condition, delta

    def calc_env_state(self, next_frame):
        self.next_frame = next_frame
        is_external_step = self.substep == 0

        if self.substep == self.frame_skip - 1:
            self.timestep += 1
        self.substep = (self.substep + 1) % self.frame_skip

        self.integrate_root_translation(next_frame)

        for agent in self.agents:
            self.calc_glove_state(agent)

    def reset(self):
        self.timestep = 0
        self.substep = 0
        self.root_facing.fill_(0)
        self.root_xz.fill_(0)
        self.done.fill_(False)
        self.foot_pos_history.fill_(1)
        self.reset_initial_frames()

        # PettingZoo
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.actions = {agent: None for agent in self.agents}
        self.observations = {agent:None for agent in self.agents}
        self.num_moves = 0

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.agents_glove_state = {agent:[[0.0,0.0,0.0] for _ in range(12)] for agent in self.agents} # 6 = self.num_characters * 3d

        # return observation

    def step(self, action): # expect action per single agent in pettingzoo

        if self.dones[self.agent_selection]:
            return self._was_done_step(action)

        agent = self.agent_selection
        # self._action_spaces[agent] = action
        # action = torch.tensor([action * self.action_scale]).to(self.device)
        self.actions[agent] = [x * self.action_scale for x in action.tolist()]
        self._cumulative_rewards[agent] = 0

        # penalty reward
        # action_penalty = self.calc_action_penalty()
        # energy_penalty = self.calc_energy_penalty(next_frame)
        # self.rewards[agent] += energy_penalty

        if self._agent_selector.is_last(): # reward
            self.num_moves += 1
            action = [self.actions[0] , self.actions[1] ]
            action = torch.tensor(action).to(self.device)
            next_frame = self.get_vae_next_frame(action)
            self.action = action
            self.calc_env_state(next_frame[:, 0])  # frame skip = 1
            self.render()
        else:
            self._clear_rewards()

        self.agent_selection = self._agent_selector.next()

    def render(self, mode="human", **kwargs):
        EnvBase.render(self)

    def close(self):
        EnvBase.close(self)

    def seed(self, seed=None):
        EnvBase.seed(self, seed)
