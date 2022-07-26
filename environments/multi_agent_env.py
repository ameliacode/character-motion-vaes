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

class AdversePlayersFightingEnv(EnvBase): # raw env
    def __init__(self,
                 rendered=True,
                 use_params=False,
                 camera_tracking=True,
                 frame_skip=1):
        # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        self.num_parallel = 2 # number of characters in this case
        self.num_characters = self.num_parallel

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
        # self.agents_foot_state = [[] for _ in range(self.num_characters)]
        self.agents_glove_state = [[] for _ in range(self.num_characters)]

        # PettingZoo
        self.possible_agents=[0,1] #players
        self.agents = self.possible_agents[:]

        self.observation_dim = (
                self.frame_dim * self.num_condition_frames + self.action_dim
        )
        high = np.inf * np.ones([self.observation_dim])
        self.observation_space = {agent: Box(-high, high, dtype=np.float32) for agent in self.possible_agents}

    # def calc_foot_state(self, agent): # player's foot and opponent target
    #     player = agent
    #     opponent = (agent + 1) % 2
    #
    #     player_foot = [self.viewer.characters.ids[player + self.foot_indices[idx]] for idx in range(len(self.foot_indices))]
    #     opponent_target = [self.viewer.characters.ids[opponent + self.target_indices[idx]] for idx in range(len(self.target_indices))]
    #
    #     foot_state = []
    #     for foot_id in player_foot: #right foot, left foot respectively
    #         foot_state = self.viewer._p.getBasePositionAndOrientation(foot_id)
    #         for target_id in opponent_target: #pelvis, abdomen respectively
    #             target_state = self.viewer._p.getBasePositionAndOrientation(target_id)
    #             relative_pos = target_state[0] - foot_state[0]
    #             if self.agents_foot_state[agent] is not None:
    #                 foot_state.append(relative_pos, relative_pos - self.agents_foot_state[agent])
    #             else:
    #                 foot_state.append((relative_pos, [0.0,0.0,0.0]))
    #
    #     self.agents_foot_state[agent] = foot_state
    #     return foot_state

    def calc_glove_state(self, agent):  # player's glove and opponent target
        player = agent
        opponent = (agent + 1) % 2

        player_glove = [self.viewer.characters.ids[player + self.glove_indices[idx]] for idx in range(len(self.glove_indices))]
        opponent_target = [self.viewer.characters.ids[opponent + self.target_indices[idx]] for idx in range(len(self.target_indices))]

        glove_state = []
        for glove_id in player_glove:  # right glove, left glove respectively
            glove_state = self.viewer._p.getBasePositionAndOrientation(glove_id)
            for target_id in opponent_target:  # pelvis, abdomen respectively
                target_state = self.viewer._p.getBasePositionAndOrientation(target_id)
                relative_pos = [target_state[0][idx] - glove_state[0][idx] for idx in range(3) ]
                if self.agents_glove_state[agent] is not None:
                    # relative_vel = [relative_pos[idx] - self.agents_foot_state[agent]]
                    glove_state.append(relative_pos, relative_pos - self.agents_glove_state[agent])
                else:
                    glove_state.append(relative_pos, [0.0, 0.0, 0.0])

        self.agents_glove_state[agent] = glove_state
        return glove_state

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
        # foot_state = self.agents_foot_state[player]
        glove_state = self.agents_glove_state[player]
        # return foot_state, glove_state
        return glove_state

    def get_defense_state(self, agent):  # opponent to player
        opponent = agent
        # foot_state = self.agents_foot_state[opponent]
        glove_state = self.agents_glove_state[opponent]

        return glove_state

    def get_observation_components(self, agent): # get observable components
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

        # for agent in self.agents:
        # #     self.calc_foot_state(agent)
        #     self.calc_glove_state(agent)

        # penalty reward
        # action_penalty = self.calc_action_penalty()
        # energy_penalty = self.calc_energy_penalty(next_frame)
        # self.rewards.add_(energy_penalty)

        # observation = self.get_observation_components()


        self.render()

        # return (
        #     torch.cat()
        # )

    def reset_agent_position(self):
        pass

    def reset(self, indices=None):
        if indices is None:
            self.timestep = 0
            self.substep = 0
            self.root_facing.fill_(0)
            self.root_xz.fill_(0)
            self.done.fill_(False)
            self.reward.fill_(0)
            self.foot_pos_history.fill_(1)
            self.reset_initial_frames()
            self.reset_agent_position()
        else:
            self.root_facing.index_fill_(dim=0, index=indices, value=0)
            self.root_xz.index_fill_(dim=0, index=indices, value=0)
            self.reward.index_fill_(dim=0, index=indices, value=0)
            self.done.index_fill_(dim=0, index=indices, value=False)
            self.foot_pos_history.index_fill_(dim=0, index=indices, value=1)
            self.reset_agent_position()

        # self.foot_state_dim = 24 # len(target_indices) * len(foot_indices) * left+right * 3D coord
        self.glove_state_dim = 6 * 6 # len(target_indices) * len(foot_indices) * left+right * 3D coord
        # self.agents_foot_state = torch.zeros((self.num_parallel, self.foot_state_dim)).to(self.device) # ??
        self.agents_glove_state = torch.zeros((self.num_parallel, self.glove_state_dim)).to(self.device) # ??

        # observation = self.get_observation_components()
        # return torch.cat(observation, dim=1)

    def step(self, action): # expect multiple actions at once (?)

        # self._action_spaces[agent] = action
        action = np.reshape(np.array(action, dtype=np.float32), (self.num_parallel, self.action_dim))
        action = torch.tensor(action * self.action_scale).to(self.device)
        next_frame = self.get_vae_next_frame(action)
        self.action = action
        state = self.calc_env_state(next_frame[:,0])

        # self.rewards[agent] += energy_penalty

        return state

    def render(self, mode="human", **kwargs):
        EnvBase.render(self)

    def close(self):
        EnvBase.close(self)

    def seed(self, seed=None):
        EnvBase.seed(self, seed)
