"""Yaw-frame velocity tracking rewards.

The stock velocity rewards use ``root_link_lin_vel_b`` / ``root_link_ang_vel_b``
which rotate with the FULL base orientation, so any pitch/roll leaks into the
xy components and penalizes tilt indirectly.  These variants rotate the world
velocity by the yaw-only quaternion instead, letting the robot lean without
being penalized.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity_yaw(
  env: "ManagerBasedRlEnv",
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Tracking reward in the YAW-ONLY base frame (pitch/roll ignored).

  Rotates the world-frame linear velocity by the inverse of the yaw quaternion
  so tilt does not leak into the xy components.  z error is penalized in the
  world frame directly (vertical motion is always bad).
  """
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  yaw_q = yaw_quat(asset.data.root_link_quat_w)
  lin_vel_yaw = quat_apply_inverse(yaw_q, asset.data.root_link_lin_vel_w)
  xy_error = torch.sum(torch.square(command[:, :2] - lin_vel_yaw[:, :2]), dim=1)
  z_error = torch.square(asset.data.root_link_lin_vel_w[:, 2])
  return torch.exp(-(xy_error + z_error) / std**2)


def track_angular_velocity_yaw(
  env: "ManagerBasedRlEnv",
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Angular velocity tracking using world-frame yaw rate.

  The commanded yaw rate is compared against the world-frame z angular
  velocity (equivalent to the yaw-frame z component since yaw rotation fixes
  the z axis).  The xy angular velocity in world frame is also penalized to
  discourage pitch/roll oscillation.
  """
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  ang_vel_w = asset.data.root_link_ang_vel_w
  z_error = torch.square(command[:, 2] - ang_vel_w[:, 2])
  xy_error = torch.sum(torch.square(ang_vel_w[:, :2]), dim=1)
  return torch.exp(-(z_error + xy_error) / std**2)
