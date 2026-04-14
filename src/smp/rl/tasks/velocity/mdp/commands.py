"""Yaw-frame variant of ``UniformVelocityCommand`` for SMP velocity tasks.

Only the debug visualization is overridden: the actual and commanded linear
velocity arrows are drawn in the yaw-only base frame so they lie horizontally
regardless of the robot's pitch/roll.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class UniformVelocityCommandYaw(UniformVelocityCommand):
  """Same command as ``UniformVelocityCommand`` but the viewer arrows are in
  the yaw-only base frame (so they stay horizontal when the robot leans)."""

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    cmds = self.command.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    # Yaw-only rotation matrix for arrow orientation.
    yaw_q = yaw_quat(self.robot.data.root_link_quat_w)
    yaw_mat_ws = matrix_from_quat(yaw_q).cpu().numpy()
    # Actual linear velocity in yaw frame.
    lin_vel_yaws = (
      quat_apply_inverse(yaw_q, self.robot.data.root_link_lin_vel_w).cpu().numpy()
    )
    # World-frame yaw rate (== z component of yaw-frame ang vel).
    ang_vel_ws = self.robot.data.root_link_ang_vel_w.cpu().numpy()

    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset

    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      yaw_mat_w = yaw_mat_ws[batch]
      cmd = cmds[batch]
      lin_vel_yaw = lin_vel_yaws[batch]
      ang_vel_w = ang_vel_ws[batch]

      # Skip if robot appears uninitialized (at origin).
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      def local_to_world(
        vec: np.ndarray,
        pos: np.ndarray = base_pos_w,
        mat: np.ndarray = yaw_mat_w,
      ) -> np.ndarray:
        return pos + mat @ vec

      # Command linear velocity arrow (blue).
      cmd_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      cmd_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([cmd[0], cmd[1], 0])) * scale
      )
      visualizer.add_arrow(
        cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015
      )

      # Command angular velocity arrow (green).
      cmd_ang_from = cmd_lin_from
      cmd_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, cmd[2]])) * scale
      )
      visualizer.add_arrow(
        cmd_ang_from, cmd_ang_to, color=(0.2, 0.6, 0.2, 0.6), width=0.015
      )

      # Actual linear velocity arrow (cyan) — drawn in yaw-only frame.
      act_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      act_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([lin_vel_yaw[0], lin_vel_yaw[1], 0]))
        * scale
      )
      visualizer.add_arrow(
        act_lin_from, act_lin_to, color=(0.0, 0.6, 1.0, 0.7), width=0.015
      )

      # Actual angular velocity arrow (light green) — yaw rate = world z.
      act_ang_from = act_lin_from
      act_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, ang_vel_w[2]])) * scale
      )
      visualizer.add_arrow(
        act_ang_from, act_ang_to, color=(0.0, 1.0, 0.4, 0.7), width=0.015
      )


@dataclass(kw_only=True)
class UniformVelocityCommandYawCfg(UniformVelocityCommandCfg):
  """Cfg that builds ``UniformVelocityCommandYaw`` instead of the stock cmd."""

  def build(self, env: "ManagerBasedRlEnv") -> UniformVelocityCommandYaw:
    return UniformVelocityCommandYaw(self, env)
