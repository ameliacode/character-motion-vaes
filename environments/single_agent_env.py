import pybullet as p
import gym
from gym.spaces import Discrete
from pathlib import Path
import numpy as np
import math
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

class PunchingPlayerEnv(TargetEnv):
    def __init__(self,
                 num_parallel,
                 device,
                 pose_vae_path,
                 rendered=True,
                 use_params=False,
                 camera_tracking=True,
                 frame_skip=1
                 ):
        self.num_parallel = num_parallel
        self.device = device
        self.current_dir = Path(__file__).resolve().parents[1]
        # self.pose_vae_path = str(self.current_dir / "vae_motion" / "models" / "posevae_c1_e6_l32.pt")
        self.pose_vae_path = pose_vae_path
        super().__init__(self.num_parallel,
                         self.device,
                         self.pose_vae_path,
                         rendered,
                         use_params,
                         camera_tracking,
                         frame_skip)

        self.root_indices = [25,18,0,7,2] # right & left shoulder, hip, right & left
        self.glove_indices = [30,23] # right and left hand

        # player_joints = self.viewer.characters.ids
        # player_glove = [[
        #     player_joints[num * 31 + self.glove_indices[idx]]
        #     for idx in range(len(self.glove_indices))
        # ] for num in range(self.num_parallel)]
        # player_root = [[
        #     player_joints[num * 31 + self.root_indices[idx]]
        #     for idx in range(len(self.root_indices))
        # ] for num in range(self.num_parallel)]
        #
        # self.player_glove = torch.tensor(player_glove).to(self.device) # number of character's glove joints
        # self.player_root = torch.tensor(player_root).to(self.device)

        self.glove_state = torch.zeros((self.num_parallel, 12)).to(self.device) # number of characters' glove state for velocity calculation
        self.target_radius = 3.0

        condition_size = self.frame_dim * self.num_condition_frames
        # 2 because we are doing cos() and sin() = target dim
        self.observation_dim = condition_size + 2 + 12
        high = np.inf * np.ones([self.observation_dim])
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.reset_condition = torch.zeros((self.num_parallel, 1)).bool().to(self.device)

    def get_observation_components(self):
        target_delta, _ = self.get_target_delta_and_angle()
        # Should be negative because going from global to local
        mat = self.get_rotation_matrix(-self.root_facing)
        delta = (mat * target_delta.unsqueeze(1)).sum(dim=2)
        condition = self.get_vae_condition(normalize=False)

        return condition, delta, self.glove_state

    def reset_initial_frames(self, frame_index=None):
        super().reset_initial_frames(frame_index)
        # to randomly face each other in marl this will be needed

    def reset(self, indices=None):
        self.reset_condition.fill_(False)
        self.glove_state = torch.zeros((self.num_parallel, 12)).to(self.device)
        return super().reset(indices)

    def calc_punch_state(self):
        target_pos = self.target
        right_glove_pos = self.glove_state[:,0:2]
        left_glove_pos = self.glove_state[:,6:8]

        # right & left punch check
        self.reset_condition = torch.nn.functional.normalize(target_pos - right_glove_pos) <= self.target_radius
                               # or torch.nn.functional.normalize(target_pos - left_glove_pos) <= self.target_radius


    def calc_glove_state(self, joints_pos): # this crashes with viewer when non rendered find other mothds..
        right_glove_state = joints_pos[:,self.glove_indices[0]]
        right_glove_vel = right_glove_state - self.glove_state[:, 0:3]

        left_glove_state = joints_pos[:,self.glove_indices[1]]
        left_glove_vel = left_glove_state - self.glove_state[:, 6:9]

        self.glove_state[:,0:3].copy_(right_glove_state)
        self.glove_state[:,3:6].copy_(right_glove_vel)
        self.glove_state[:,6:9].copy_(left_glove_state)
        self.glove_state[:,9:12].copy_(left_glove_vel)

    def calc_facing_dir(self, joints_pos): # try to use "pose" variable, visualize two methods later

        shoulders = [joints_pos[:,self.root_indices[idx]] * 0.3048 for idx in range(0,2)]
        hip = joints_pos[:, self.root_indices[2]]* 0.3048
        left_shoulder = shoulders[0] - hip
        right_shoulder = shoulders[1] - hip

        cross_product = torch.cross(left_shoulder, right_shoulder, dim=1)
        normal_vector = torch.nn.functional.normalize(cross_product)

        target_delta, _ = self.get_target_delta_and_angle()
        y = normal_vector[:,1] - target_delta[:,1]
        x = normal_vector[:,0] - target_delta[:,0]
        radian = torch.atan2(y,x)

        if self.is_rendered:
            cross_joint = joints_pos[:,0][0]*0.3048 + normal_vector[:][0]*0.3048
            root_joint = joints_pos[:,0][0]*0.3048

            self.viewer._p.addUserDebugLine(
                lineFromXYZ=root_joint.tolist(),
                lineToXYZ=cross_joint.tolist(),
                lineColorRGB=(0,0,255),
                lineWidth=2,
                lifeTime=1,
            )

        reward = torch.cos(radian) * 100
        reward = reward.view(self.num_parallel,1)
        self.reward.add_(reward)

    def calc_env_state(self, next_frame):
        self.next_frame = next_frame
        is_external_step = self.substep == 0

        if self.substep == self.frame_skip - 1:
            self.timestep += 1
        self.substep = (self.substep + 1) % self.frame_skip

        self.integrate_root_translation(next_frame)

        progress = self.calc_progress_reward()

        target_dist = -self.linear_potential
        target_is_close = target_dist < 4.0

        if is_external_step:
            self.reward.copy_(progress)
        else:
            self.reward.add_(progress)

        x, y, z = extract_joints_xyz(next_frame, *self.joint_indices, dim=1)
        joints_xy = self.root_xz.unsqueeze(1) + torch.stack((x, y), dim=-1)
        joints_pos = torch.cat((joints_xy, z.unsqueeze(-1)), dim=-1)

        self.calc_glove_state(joints_pos)
        self.calc_facing_dir(joints_pos)

        self.reward.add_(target_is_close.float() * 20.0)

        energy_penalty = self.calc_energy_penalty(next_frame)
        self.reward.add_(energy_penalty)

        if target_is_close.any():
            self.calc_punch_state()
            if self.reset_condition.any():
                reset_indices = self.parallel_ind_buf.masked_select( # used for reward-based early termination
                    target_is_close.squeeze(1)
                )
                self.reward.add_(1000)
                self.reset_target(indices=reset_indices)

        obs_component = self.get_observation_components()
        self.done.fill_(self.timestep >= self.max_timestep)
        self.render()

        return (
            torch.cat(obs_component, dim=1),
            self.reward,
            self.done,
            {"reset":self.timestep >= self.max_timestep},
        )