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
import random

from scipy.spatial.transform import *
import scipy.ndimage.filters as filters
import pytorch3d

try:
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from mocap_envs import *
except:
    from .mocap_envs import *

class CustomTargetEnv(TargetEnv):
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

        self.target_direction_buf = torch.zeros((self.num_parallel, 1)).to(self.device)
        self.target_speed_buf = torch.zeros((self.num_parallel, 1)).to(self.device)
        self.target_speed = torch.ones((self.num_parallel, 1)).to(self.device)

    def _get_target_delta_and_angle(self): # implement from deepmimic
        target_delta = self.target - self.root_xz
        target_angle = (
            torch.atan2(-target_delta[:,1], target_delta[:,0]).unsqueeze(1)
        )
        return target_delta, target_angle

    def calc_progress_reward_joystick(self): # target heading (from joystick env)
        _, target_angle = self.get_target_delta_and_angle()
        direction_reward = target_angle.cos().add(-1)

        return direction_reward.exp()

    def calc_progress_reward_deepmimic(self): # target heading (from deepmimic)
        target_delta, target_angle = self.get_target_delta_and_angle()
        target_dir = torch.cat((torch.cos(target_angle[:]), -torch.sin(target_angle[:])), dim=1).unsqueeze(1)
        avg_vel = (self.displacement / (1 / self.data_fps)).unsqueeze(1)
        avg_speed = torch.diag(torch.tensordot(target_dir, avg_vel, dims=([1,2],[2,1]))).unsqueeze(1)

        vel_err = torch.cat((self.target_speed - avg_speed, torch.zeros((self.num_parallel, 1)).to(self.device)), 1)
        vel_err = torch.max(vel_err, 1).values.unsqueeze(1)
        vel_reward = torch.exp(-1.0 * vel_err * vel_err)

        self.calc_potential()
        # print(self.root_xz, self.target, self.linear_potential)
        return vel_reward

    def calc_pos_reward(self): # target heading (from ACE - location)
        pos_err_scale = 0.5
        pos_diff, _ = self.get_target_delta_and_angle()
        pos_diff = pos_diff.unsqueeze(1)
        pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
        pos_reward = torch.exp(-pos_err_scale * pos_err)

        return pos_reward, pos_err

    def calc_vel_reward(self):
        vel_err_scale = 0.25
        target_delta, _ = self.get_target_delta_and_angle()
        target_dir = torch.nn.functional.normalize(target_delta, dim=-1).unsqueeze(1)
        avg_vel = (self.displacement / (1 / self.data_fps)).unsqueeze(1) # root vel
        avg_speed = torch.sum(target_dir * avg_vel, dim=-1) # tar_dir_speed

        tar_dir_speed = avg_speed
        tar_vel_err = self.target_speed - tar_dir_speed
        tar_vel_err = torch.clamp_min(tar_vel_err, 0.0)
        vel_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err))
        speed_mask = tar_dir_speed <= 0
        vel_reward[speed_mask] = 0

        return vel_reward, target_dir, avg_speed, avg_vel

    def calc_facing_reward(self, target_dir):
        facing_dir = torch.cat((torch.cos(-self.root_facing), -torch.sin(-self.root_facing)), dim=1).unsqueeze(1)
        facing_err = torch.sum(target_dir * facing_dir, dim=-1)
        facing_reward = torch.clamp_min(facing_err, 0.0)

        return facing_reward

    def calc_location_reward(self): # target heading (from ACE - location)
        dist_threshold = 0.5
        pos_reward_w = 0.5
        pos_reward, pos_err = self.calc_pos_reward()
        vel_reward_w = 0.4
        vel_reward, target_dir, _, _ = self.calc_vel_reward()
        face_reward_w = 0.1
        facing_reward = self.calc_facing_reward(target_dir)

        dist_mask = pos_err < dist_threshold
        facing_reward[dist_mask] = 1.0
        vel_reward[dist_mask] = 1.0

        self.calc_potential()

        return pos_reward_w * pos_reward + vel_reward_w * vel_reward + face_reward_w * facing_reward

    def calc_heading_reward(self): # target heading (from ACE - steering as heading.py )
        vel_err_scale = 0.25
        tangent_err_w = 0.1

        dir_reward_w = 0.7
        facing_reward_w = 0.3

        _, target_dir, tar_dir_speed, root_vel = self.calc_vel_reward()
        tar_dir_vel = tar_dir_speed.unsqueeze(-1) * target_dir
        tangent_vel = root_vel - tar_dir_vel
        tangent_speed = torch.sum(tangent_vel, dim=-1)

        tar_vel_err = self.target_speed - tar_dir_speed
        tangent_vel_err = tangent_speed

        dir_reward = torch.exp(-vel_err_scale*(tar_vel_err * tar_vel_err + tangent_err_w * tangent_vel_err * tangent_vel_err))

        speed_mask = tar_dir_speed <= 0
        dir_reward[speed_mask] = 0

        facing_reward = self.calc_facing_reward(target_dir)

        self.calc_potential()

        return dir_reward_w * dir_reward + facing_reward_w * facing_reward


    def calc_progress_reward(self):
        return self.calc_location_reward()

class CustomPunchingEnv(TargetEnv):
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

        # self.root_indices = [25,18,0,7,2] # right & left shoulder, hip, right & left hip # CMU
        self.root_indices = [18,14,0,5,1]
        # self.glove_indices = [30,23] # right and left hand

        self.glove_state = torch.zeros((self.num_parallel, 12)).to(self.device) # number of characters' glove state for velocity calculation
        self.target_radius = 3.0

        condition_size = self.frame_dim * self.num_condition_frames
        # 2 because we are doing cos() and sin() = target dim
        self.observation_dim = condition_size + 2 + 12
        high = np.inf * np.ones([self.observation_dim])
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.reset_condition = torch.zeros((self.num_parallel, 1)).bool().to(self.device)
        self.alpha = 1.0

    def get_observation_components(self):
        target_delta, _ = self.get_target_delta_and_angle()
        # Should be negative because going from global to local
        mat = self.get_rotation_matrix(-self.root_facing)
        delta = (mat * target_delta.unsqueeze(1)).sum(dim=2)
        condition = self.get_vae_condition(normalize=False) # 1 * 375

        return condition, delta, self.glove_state

    def reset_initial_frames(self, frame_index=None):
        super().reset_initial_frames(frame_index)
        # to randomly face each other in marl this will be needed

    def reset(self, indices=None):
        self.reset_condition.fill_(False)
        self.glove_state = torch.zeros((self.num_parallel, 12)).to(self.device)
        self.alpha = 1.0
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

    def calc_facing_dir(self, joints_pos):  # try to use "pose" variable, visualize two methods later
        shoulders = [joints_pos[:, self.root_indices[idx]] for idx in range(0, 2)]
        hips = [joints_pos[:, self.root_indices[idx]] for idx in range(3, 5)]
        shoulder = shoulders[1] - shoulders[0]
        hip = hips[1] - hips[0]
        across = torch.Tensor(shoulder+hip)
        # across = across / np.sqrt((across ** 2)).sum(axis=-1)[...,np.newaxis]

        cross_product = torch.cross(torch.Tensor([[across[0][0], across[0][1], 0.0]]), torch.Tensor([[0,0,1]]), dim=1)
        # normal_vector = cross_product / np.sqrt((cross_product ** 2).sum(axis=-1))[...,np.newaxis]
        normal_vector = torch.nn.functional.normalize(cross_product)

        # target delta: root_xz -> target
        # normal vector: root_xz -> root_xz + normal_vec

        root_joint = torch.cat((self.root_xz, torch.Tensor([[0.0]])), dim=1)
        cross_joint = root_joint[:][0] * 0.3048 - normal_vector[:][0]

        target_delta, target_angle = self.get_target_delta_and_angle()
        cross_joint = torch.Tensor([[cross_joint[0], cross_joint[1]]])

        y = cross_joint[:, 1] - target_delta[:, 1]
        x = cross_joint[:, 0] - target_delta[:, 0]
        radian = torch.atan2(y, x)

        if self.is_rendered:
            # root_joint = joints_pos[:, 0][0] * 0.3048

            root_joint = root_joint.tolist()[0]
            across = across.tolist()[0]
            cross_joint = cross_joint.tolist()[0]

            self.viewer._p.addUserDebugLine( # CROSS
                lineFromXYZ=[root_joint[0] * 0.3048, root_joint[1] * 0.3048, 0],
                lineToXYZ=[cross_joint[0], cross_joint[1], 0.0],
                lineColorRGB=(0,255,255),
                lineWidth=2,
                lifeTime=0.05,
            )
            self.viewer._p.addUserDebugLine( # UP
                lineFromXYZ=[root_joint[0] * 0.3048, root_joint[1] * 0.3048, 0],
                lineToXYZ=[root_joint[0] * 0.3048, root_joint[1] * 0.3048, 1.0],
                lineColorRGB=(0,0,255),
                lineWidth=2,
                lifeTime=0.05,
            )
            self.viewer._p.addUserDebugLine( # ACROSS
                lineFromXYZ=[root_joint[0] * 0.3048, root_joint[1] * 0.3048, 0],
                lineToXYZ=[root_joint[0] * 0.3048 + across[0], root_joint[1] * 0.3048 + across[1], 0],
                lineColorRGB=(0,0,255),
                lineWidth=2,
                lifeTime=0.05,
            )
            self.viewer._p.addUserDebugLine(
                lineFromXYZ=[root_joint[0] * 0.3048, root_joint[1] * 0.3048, 0],
                lineToXYZ=[self.target.tolist()[0][0] * 0.3048, self.target.tolist()[0][1] * 0.3048, 0],
                lineColorRGB=(0, 255, 0),
                lineWidth=2,
                lifeTime=0.05,
            )

            # self.viewer._p.addUserDebugLine(
            #     lineFromXYZ=shoulders[1].tolist()[0],
            #     lineToXYZ=[0.0,0.0,0.0],
            #     lineColorRGB=(0, 255, 0),
            #     lineWidth=2,
            #     lifeTime=0.05,
            # )

        reward = torch.cos(radian) * 100
        reward = reward.view(self.num_parallel, 1)
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

        self.render()
        # x, y, z = extract_joints_xyz(next_frame, *self.joint_indices, dim=1)
        # joints_xy = self.root_xz.unsqueeze(1) + torch.stack((x, y), dim=-1)
        # joints_pos = torch.cat((joints_xy, z.unsqueeze(-1)), dim=-1)
        # print(joints_pos)

        joints_pos = self.viewer.joint_xyzs

        # self.calc_glove_state(joints_pos)
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
        # self.render()

        return (
            torch.cat(obs_component, dim=1),
            self.reward,
            self.done,
            {"reset":self.timestep >= self.max_timestep},
        )


class CustomHoppingEnv(TargetEnv):
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

        self.root_indices = [25, 18, 0, 7, 2]
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
        return super().reset(indices)

    def calc_facing_dir(self, joints_pos):  # try to use "pose" variable, visualize two methods later

        shoulders = [joints_pos[:, self.root_indices[idx]] for idx in range(0, 2)]
        hips = [joints_pos[:, self.root_indices[idx]] for idx in range(3, 5)]
        shoulder = shoulders[1] - shoulders[0]  # left - right
        hip = hips[1] - hips[0]
        across = torch.Tensor(shoulder + hip)
        across = across / np.sqrt((across ** 2)).sum(axis=-1)[..., np.newaxis]

        forward = filters.gaussian_filter1d(
            np.cross(across, np.array([[0, 0, 1]])), 20, axis=0, mode="nearest"
        )
        forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]
        normal_vector = forward

        # target delta: root_xz -> target
        # normal vector: root_xz -> root_xz + normal_vec

        root_joint = torch.cat((self.root_xz, torch.Tensor([[0.0]])), dim=1)
        cross_joint = root_joint[:][0] * 0.3048 + normal_vector[:][0]

        target_delta, target_angle = self.get_target_delta_and_angle()
        cross_joint = torch.Tensor([[cross_joint[0], cross_joint[1]]])

        y = cross_joint[:, 1] - target_delta[:, 1]
        x = cross_joint[:, 0] - target_delta[:, 0]
        radian = torch.atan2(y, x)

        if self.is_rendered:
            # root_joint = joints_pos[:, 0][0] * 0.3048

            root_joint = root_joint.tolist()[0]
            across = across.tolist()[0]
            cross_joint = cross_joint.tolist()[0]

            self.viewer._p.addUserDebugLine(  # CROSS
                lineFromXYZ=[root_joint[0] * 0.3048, root_joint[1] * 0.3048, 0],
                lineToXYZ=[cross_joint[0], cross_joint[1], 0.0],
                lineColorRGB=(0, 255, 255),
                lineWidth=2,
                lifeTime=0.05,
            )
            self.viewer._p.addUserDebugLine(  # UP
                lineFromXYZ=[root_joint[0] * 0.3048, root_joint[1] * 0.3048, 0],
                lineToXYZ=[root_joint[0] * 0.3048, root_joint[1] * 0.3048, 1.0],
                lineColorRGB=(0, 0, 255),
                lineWidth=2,
                lifeTime=0.05,
            )
            self.viewer._p.addUserDebugLine(  # ACROSS
                lineFromXYZ=[root_joint[0] * 0.3048, root_joint[1] * 0.3048, 0],
                lineToXYZ=[root_joint[0] * 0.3048 + across[0], root_joint[1] * 0.3048 + across[1], 0],
                lineColorRGB=(0, 0, 255),
                lineWidth=2,
                lifeTime=0.05,
            )
            self.viewer._p.addUserDebugLine(
                lineFromXYZ=[root_joint[0] * 0.3048, root_joint[1] * 0.3048, 0],
                lineToXYZ=[self.target.tolist()[0][0] * 0.3048, self.target.tolist()[0][1] * 0.3048, 0],
                lineColorRGB=(0, 255, 0),
                lineWidth=2,
                lifeTime=0.05,
            )

        # reward = torch.cos(radian) * 100
        # reward = reward.view(self.num_parallel, 1)
        # self.reward.add_(reward)

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

        self.calc_facing_dir(joints_pos)

        self.reward.add_(target_is_close.float() * 20.0)

        energy_penalty = self.calc_energy_penalty(next_frame)
        self.reward.add_(energy_penalty)

        if target_is_close.any():

            if self.reset_condition.any():
                reset_indices = self.parallel_ind_buf.masked_select(  # used for reward-based early termination
                    target_is_close.squeeze(1)
                )
                self.reward.add_(1000)
                self.reset_target(indices=reset_indices)

        obs_component = self.get_observation_components()
        self.done.fill_(self.timestep >= self.max_timestep)
        self.render()
        # assert joints_pos == self.viewer.joint_xyzs

        return (
            torch.cat(obs_component, dim=1),
            self.reward,
            self.done,
            {"reset": self.timestep >= self.max_timestep},
        )

#
# def calc_facing_dir(self, joints_pos):
#     self.root_indices = [18,14,0,5,1]
#     shoulders = [joints_pos[:, self.root_indices[idx]] for idx in range(0, 2)]
#     hips = [joints_pos[:, self.root_indices[idx]] for idx in range(3, 5)]
#     shoulder = shoulders[0] - shoulders[1]  # left - right
#     hip = hips[0] - hips[1]
#     across = torch.Tensor(shoulder + hip)
#     across = across / np.sqrt((across ** 2)).sum(axis=-1)[..., np.newaxis]
#
#     forward = filters.gaussian_filter1d(
#         np.cross(across, np.array([[0, 0, 1]])), 20, axis=0, mode="nearest"
#     )
#     forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]
#     normal_vector = forward
#
#     # target delta: root_xz -> target
#     # normal vector: root_xz -> root_xz + normal_vec
#
#     root_joint = torch.cat((self.root_xz, torch.Tensor([[0.0]])), dim=1)
#     cross_joint = root_joint[:][0] * 0.3048 + normal_vector[:][0]
#
#     target_delta, target_angle = self.get_target_delta_and_angle()
#     cross_joint = torch.Tensor([[cross_joint[0], cross_joint[1]]])
#
#     y = cross_joint[:, 1] - target_delta[:, 1]
#     x = cross_joint[:, 0] - target_delta[:, 0]
#     radian = torch.atan2(y, x)
#
#     if self.is_rendered:
#         # root_joint = joints_pos[:, 0][0] * 0.3048
#
#         root_joint = root_joint.tolist()[0]
#         across = across.tolist()[0]
#         cross_joint = cross_joint.tolist()[0]
#
#         self.viewer._p.addUserDebugLine(  # CROSS
#             lineFromXYZ=[root_joint[0] * 0.3048, root_joint[1] * 0.3048, 0],
#             lineToXYZ=[cross_joint[0], cross_joint[1], 0.0],
#             lineColorRGB=(0, 255, 255),
#             lineWidth=2,
#             lifeTime=0.05,
#         )
#         self.viewer._p.addUserDebugLine(  # UP
#             lineFromXYZ=[root_joint[0] * 0.3048, root_joint[1] * 0.3048, 0],
#             lineToXYZ=[root_joint[0] * 0.3048, root_joint[1] * 0.3048, 1.0],
#             lineColorRGB=(0, 0, 255),
#             lineWidth=2,
#             lifeTime=0.05,
#         )
#         self.viewer._p.addUserDebugLine(  # ACROSS
#             lineFromXYZ=[root_joint[0] * 0.3048, root_joint[1] * 0.3048, 0],
#             lineToXYZ=[root_joint[0] * 0.3048 + across[0], root_joint[1] * 0.3048 + across[1], 0],
#             lineColorRGB=(0, 0, 255),
#             lineWidth=2,
#             lifeTime=0.05,
#         )
#         self.viewer._p.addUserDebugLine(
#             lineFromXYZ=[root_joint[0] * 0.3048, root_joint[1] * 0.3048, 0],
#             lineToXYZ=[self.target.tolist()[0][0] * 0.3048, self.target.tolist()[0][1] * 0.3048, 0],
#             lineColorRGB=(0, 255, 0),
#             lineWidth=2,
#             lifeTime=0.05,
#         )
#
#     # reward = torch.cos(radian) * 100
#     # reward = reward.view(self.num_parallel, 1)
#     # self.reward.add_(reward)