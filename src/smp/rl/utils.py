"""Utilities for SMP RL: denoiser loader, diff-normalizer, and feature buffer.

Grouped in one module for a cleaner package layout.  All of these are used
by the SMP guidance reward and the associated startup/reset events.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  subtract_frame_transforms,
  yaw_quat,
)

from smp.pretrain.model import DiffusionDenoiser
from smp.pretrain.scheduler import DDPMScheduler

# =============================================================================
# Denoiser loader
# =============================================================================


def load_denoiser(
  ckpt_path: str,
  device: torch.device | str,
) -> tuple[DiffusionDenoiser, DDPMScheduler, torch.Tensor, torch.Tensor, int, int]:
  """Load a frozen pretrained denoiser checkpoint.

  Returns ``(model, scheduler, q_low, q_high, feature_dim, window_size)``.
  """
  device = torch.device(device)

  ckpt: dict[str, Any] = torch.load(ckpt_path, map_location=device, weights_only=False)
  cfg = ckpt["cfg"]
  feature_dim = int(cfg["feature_dim"])
  window_size = int(cfg["window_size"])

  model = DiffusionDenoiser(
    feature_dim=feature_dim,
    window_size=window_size,
    d_model=int(cfg.get("d_model", 256)),
    nhead=int(cfg.get("nhead", 8)),
    num_layers=int(cfg.get("num_layers", 2)),
    dim_feedforward=int(cfg.get("dim_feedforward", 1024)),
    dropout=float(cfg.get("dropout", 0.0)),
  ).to(device)
  state = ckpt.get("model_ema") or ckpt["model"]
  model.load_state_dict(state)
  model.eval()
  model.requires_grad_(False)

  scheduler = DDPMScheduler(
    num_timesteps=int(cfg.get("num_timesteps", 50)),
  ).to(device)

  q_low = torch.from_numpy(np.asarray(ckpt["q_low"], dtype=np.float32)).to(device)
  q_high = torch.from_numpy(np.asarray(ckpt["q_high"], dtype=np.float32)).to(device)

  return model, scheduler, q_low, q_high, feature_dim, window_size


# =============================================================================
# Diff-normalizer (MimicKit style)
# =============================================================================


class DiffNormalizer:
  """Count-based running mean, one scalar per diffusion timestep.

  Equal weighting across all observed samples (mirrors MimicKit's
  ``DiffNormalizer``) so the normalizer naturally freezes as the sample
  count grows, giving a stable reference scale for SDS MSE values instead
  of a moving EMA target that drifts with the policy.
  """

  def __init__(
    self,
    num_timesteps: int,
    device: torch.device,
    min_value: float = 1e-4,
    max_count: int = 100_000_000,
  ) -> None:
    self.min_value = min_value
    self.max_count = max_count
    self.mean = torch.ones(num_timesteps, device=device)
    self.count = torch.zeros(num_timesteps, device=device, dtype=torch.long)

  def update_and_normalize(self, t: int, mse_per_env: torch.Tensor) -> torch.Tensor:
    """Record a per-env batch of MSE values for timestep ``t`` and return
    ``mse_per_env`` divided by the running mean at ``t``."""
    if self.count[t] > self.max_count:
      # Freeze once enough samples have been seen — the mean is stable and
      # further updates would barely move it (and risk count overflow).
      return mse_per_env / self.mean[t].clamp(min=self.min_value)
    n = mse_per_env.numel()
    batch_mean = mse_per_env.mean()
    old_count = self.count[t].item()
    new_count = old_count + n
    if old_count == 0:
      self.mean[t] = batch_mean
    else:
      w_old = old_count / new_count
      w_new = n / new_count
      self.mean[t] = w_old * self.mean[t] + w_new * batch_mean
    self.count[t] = new_count
    return mse_per_env / self.mean[t].clamp(min=self.min_value)


# =============================================================================
# Pelvis-anchored rolling feature buffer
# =============================================================================


class PelvisAnchoredFeatureBuffer:
  """Per-env rolling buffer of (root_pos_w, root_quat_w, root_lin_vel_w,
  root_ang_vel_w, joint_pos, joint_vel).

  Calling :meth:`compute_features` produces tensors with the same layout as
  the offline NPZ ``windows`` rows: ``[anchor_pos_b(3), anchor_ori_6d(6),
  base_lin_vel_b(3), base_ang_vel_b(3), joint_pos(J), joint_vel(J)]``.
  """

  def __init__(
    self,
    num_envs: int,
    window_size: int,
    num_joints: int,
    device: torch.device | str,
  ) -> None:
    self.num_envs = num_envs
    self.window_size = window_size
    self.num_joints = num_joints
    self.device = torch.device(device)

    self.root_pos_w = torch.zeros(num_envs, window_size, 3, device=self.device)
    self.root_quat_w = torch.zeros(num_envs, window_size, 4, device=self.device)
    self.root_quat_w[..., 0] = 1.0
    self.root_lin_vel_w = torch.zeros(num_envs, window_size, 3, device=self.device)
    self.root_ang_vel_w = torch.zeros(num_envs, window_size, 3, device=self.device)
    self.joint_pos = torch.zeros(num_envs, window_size, num_joints, device=self.device)
    self.joint_vel = torch.zeros(num_envs, window_size, num_joints, device=self.device)

  def reset(
    self,
    env_ids: torch.Tensor,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    root_lin_vel_w: torch.Tensor,
    root_ang_vel_w: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    """Fill all W slots of ``env_ids`` with the given current observation."""
    if env_ids.numel() == 0:
      return
    self.root_pos_w[env_ids] = root_pos_w
    self.root_quat_w[env_ids] = root_quat_w
    self.root_lin_vel_w[env_ids] = root_lin_vel_w
    self.root_ang_vel_w[env_ids] = root_ang_vel_w
    self.joint_pos[env_ids] = joint_pos
    self.joint_vel[env_ids] = joint_vel

  def update(
    self,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    root_lin_vel_w: torch.Tensor,
    root_ang_vel_w: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    """Shift left by one and append the new frame at index W-1."""
    self.root_pos_w = torch.roll(self.root_pos_w, shifts=-1, dims=1)
    self.root_quat_w = torch.roll(self.root_quat_w, shifts=-1, dims=1)
    self.root_lin_vel_w = torch.roll(self.root_lin_vel_w, shifts=-1, dims=1)
    self.root_ang_vel_w = torch.roll(self.root_ang_vel_w, shifts=-1, dims=1)
    self.joint_pos = torch.roll(self.joint_pos, shifts=-1, dims=1)
    self.joint_vel = torch.roll(self.joint_vel, shifts=-1, dims=1)
    self.root_pos_w[:, -1] = root_pos_w
    self.root_quat_w[:, -1] = root_quat_w
    self.root_lin_vel_w[:, -1] = root_lin_vel_w
    self.root_ang_vel_w[:, -1] = root_ang_vel_w
    self.joint_pos[:, -1] = joint_pos
    self.joint_vel[:, -1] = joint_vel

  def compute_features(self) -> torch.Tensor:
    """Return (num_envs, window_size, 3+6+3+3+J+J) features.

    Layout (matches ``scripts/csv_to_npz.py::_compute_windows``):
      [0:3]         motion_anchor_pos_b   anchor at frame t in frame-0 yaw frame
      [3:9]         motion_anchor_ori_b   first 2 cols of rotation matrix (6D)
      [9:12]        base_lin_vel_b        linear velocity in frame-0 yaw frame
      [12:15]       base_ang_vel_b        angular velocity in frame-0 yaw frame
      [15:15+J]     joint_pos             raw joint positions
      [15+J:15+2J]  joint_vel             raw joint velocities
    """
    N = self.num_envs
    W = self.window_size

    anchor_pos_t = self.root_pos_w  # (N, W, 3)
    anchor_quat_t = self.root_quat_w  # (N, W, 4)
    anchor_pos_0 = anchor_pos_t[:, 0:1, :].expand(N, W, 3).clone()
    anchor_pos_0[..., 2] = 0.0
    anchor_quat_0: torch.Tensor = yaw_quat(anchor_quat_t[:, 0])[:, None, :].expand(
      N, W, 4
    )

    m_pos, m_quat = subtract_frame_transforms(
      anchor_pos_0.reshape(-1, 3),
      anchor_quat_0.reshape(-1, 4),
      anchor_pos_t.reshape(-1, 3),
      anchor_quat_t.reshape(-1, 4),
    )
    m_pos = m_pos.reshape(N, W, 3)
    m_ori_6d = matrix_from_quat(m_quat)[..., :2].reshape(N, W, 6)

    # Velocities in frame-0 yaw frame.
    lin_vel_b = quat_apply_inverse(
      anchor_quat_0.reshape(-1, 4),
      self.root_lin_vel_w.reshape(-1, 3),
    ).reshape(N, W, 3)
    ang_vel_b = quat_apply_inverse(
      anchor_quat_0.reshape(-1, 4),
      self.root_ang_vel_w.reshape(-1, 3),
    ).reshape(N, W, 3)

    return torch.cat(
      [m_pos, m_ori_6d, lin_vel_b, ang_vel_b, self.joint_pos, self.joint_vel],
      dim=-1,
    )
