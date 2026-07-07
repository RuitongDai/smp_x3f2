"""将 SMP feature window 还原成机器人世界坐标状态。

这个文件主要用于：
1. 可视化 diffusion 生成的 motion window。
2. GSI 初始化时，把生成的 feature window 写回仿真状态。
3. 从局部坐标系 feature 还原 pelvis、关节和末端点轨迹。

当前 X3_F2 feature 单帧布局为 38 维：

  root_pos        (3)              xy 在最后一帧 heading-inv 坐标系下，
                                    z 保持世界坐标
  root_rot        (6)              heading_inv(T) ⊗ root_quat[t] 的 6D tan-norm
  joint_pos       (14)             X3_F2 的 14 个关节角
  ee_pos          (3*3=9)          左脚、右脚、torso_link 的局部末端位置
  root_lin_vel    (3)              最后一帧 heading-inv 坐标系下的根线速度
  root_ang_vel    (3)              最后一帧 heading-inv 坐标系下的根角速度

注意：
所有空间量都锚定到窗口最后一帧的 yaw-only 局部坐标系。
"""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_conjugate,
  quat_from_matrix,
  quat_mul,
  yaw_quat,
)

NUM_JOINTS = 14

EE_BODY_NAMES: tuple[str, ...] = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "torso_link",
)
NUM_EE = len(EE_BODY_NAMES)


def slice_features(frame: torch.Tensor) -> dict[str, torch.Tensor]:
  """将一帧或一批 feature 按字段切开。

  feature 布局：
    [0:3]                   root_pos
    [3:9]                   root_rot
    [9:9+J]                 joint_pos
    [9+J:9+J+E*3]           ee_pos
    [9+J+E*3:12+J+E*3]      root_lin_vel
    [12+J+E*3:15+J+E*3]     root_ang_vel

  当前 X3_F2：
    J = 14
    E = 3
    feature_dim = 38
  """
  J = NUM_JOINTS
  E = NUM_EE
  expected = 3 + 6 + J + E * 3 + 3 + 3

  if frame.shape[-1] != expected:
    msg = f"expected feature_dim={expected}; got {frame.shape[-1]}"
    raise ValueError(msg)

  joint_pos_end = 9 + J
  ee_pos_end = joint_pos_end + E * 3
  lin_vel_end = ee_pos_end + 3
  ang_vel_end = lin_vel_end + 3

  return {
    "root_pos": frame[..., 0:3],
    "root_rot": frame[..., 3:9],
    "joint_pos": frame[..., 9:joint_pos_end],
    "ee_pos": frame[..., joint_pos_end:ee_pos_end],
    "root_lin_vel": frame[..., ee_pos_end:lin_vel_end],
    "root_ang_vel": frame[..., lin_vel_end:ang_vel_end],
  }


def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
  """将 6D tan-norm 表示还原成 3x3 旋转矩阵。

  csv_to_npz_x3f2.py 中保存的是旋转矩阵的第 0 列和第 2 列：
    d6 = [col0, col2]

  这里通过 Gram-Schmidt 重新正交化，减少数值误差。
  """
  col0 = d6[..., :3]
  col2 = d6[..., 3:6]

  col0 = torch.nn.functional.normalize(col0, dim=-1)
  col2 = col2 - (col0 * col2).sum(dim=-1, keepdim=True) * col0
  col2 = torch.nn.functional.normalize(col2, dim=-1)

  # 右手坐标系：col0 × col1 = col2，所以 col1 = col2 × col0。
  col1 = torch.cross(col2, col0, dim=-1)

  return torch.stack([col0, col1, col2], dim=-1)


def rot6d_to_quat(d6: torch.Tensor) -> torch.Tensor:
  """将 6D 旋转表示转换成 wxyz 四元数。"""
  return quat_from_matrix(rot6d_to_matrix(d6))


def window_to_pelvis_trajectory(
  window: torch.Tensor,
  anchor_pelvis_pos_w: torch.Tensor,
  anchor_pelvis_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """将一个 feature window 还原成世界坐标系下的 pelvis 轨迹和关节角。

  Args:
    window: 形状为 (W, F) 的反归一化 feature window。
    anchor_pelvis_pos_w: 最后一帧 pelvis 的世界坐标位置，形状为 (3,)。
    anchor_pelvis_quat_w: 最后一帧 pelvis 的世界坐标四元数，形状为 (4,)。

  Returns:
    pelvis_pos_w:  形状为 (W, 3) 的 pelvis 世界坐标位置。
    pelvis_quat_w: 形状为 (W, 4) 的 pelvis 世界坐标四元数。
    joint_pos:     形状为 (W, 14) 的 X3_F2 关节角。
  """
  parts = slice_features(window)
  root_pos_local = parts["root_pos"]
  root_rot_6d = parts["root_rot"]
  W = window.shape[0]

  anchor_pelvis_pos_w = anchor_pelvis_pos_w.to(window)
  anchor_pelvis_quat_w = anchor_pelvis_quat_w.to(window)

  # 只取最后一帧 pelvis 的 yaw，作为整个 window 的 heading 坐标系。
  yaw_T = yaw_quat(anchor_pelvis_quat_w[None]).squeeze(0)

  # 还原 root_pos：
  # xy 从 heading 坐标系转回世界坐标，z 直接使用 feature 里的世界高度。
  local_xy = root_pos_local.clone()
  local_xy[..., 2] = 0.0

  world_offset_xy = quat_apply(yaw_T[None].expand(W, 4), local_xy)
  pelvis_pos_w = world_offset_xy + anchor_pelvis_pos_w[None, :3]
  pelvis_pos_w = pelvis_pos_w.clone()
  pelvis_pos_w[..., 2] = root_pos_local[..., 2]

  # 还原 root_rot：
  # root_quat_w[t] = yaw_T ⊗ root_rot_local[t]
  root_rot_local_quat = rot6d_to_quat(root_rot_6d)
  pelvis_quat_w = quat_mul(yaw_T[None].expand(W, 4), root_rot_local_quat)

  return pelvis_pos_w, pelvis_quat_w, parts["joint_pos"]


def window_to_ee_trajectories(
  window: torch.Tensor,
  pelvis_pos_w: torch.Tensor,
  pelvis_quat_w: torch.Tensor,
) -> torch.Tensor:
  """将 feature window 中的末端点位置还原到世界坐标系。

  feature 中的 ee_pos 是：
    ee_pos_local = heading_inv(T) * (ee_pos_w[t] - pelvis_pos_w[t])

  所以还原时需要：
    1. 使用最后一帧 pelvis yaw 得到 yaw_T。
    2. 将 ee_pos_local 转回世界方向。
    3. 加上每一帧 pelvis_pos_w。

  Returns:
    形状为 (W, NUM_EE, 3) 的末端点世界坐标。
  """
  parts = slice_features(window)
  W = window.shape[0]
  E = NUM_EE

  ee_pos_local = parts["ee_pos"].reshape(W, E, 3)

  yaw_T = yaw_quat(pelvis_quat_w[-1:])
  yaw_T_E = yaw_T.expand(W, 4)[:, None, :].expand(W, E, 4).reshape(-1, 4)

  ee_offset_w = quat_apply(yaw_T_E, ee_pos_local.reshape(-1, 3)).reshape(W, E, 3)

  return ee_offset_w + pelvis_pos_w[:, None, :]


def tan_norm_from_quat(quat: torch.Tensor) -> torch.Tensor:
  """将 wxyz 四元数转换成 6D tan-norm 表示。

  这里使用旋转矩阵的第 0 列和第 2 列：
    [col0, col2]
  """
  mat = matrix_from_quat(quat)
  col0 = mat[..., :, 0]
  col2 = mat[..., :, 2]

  return torch.cat([col0, col2], dim=-1)


def heading_inv_quat(quat: torch.Tensor) -> torch.Tensor:
  """返回世界坐标四元数对应的 yaw-only 逆旋转。"""
  return quat_conjugate(yaw_quat(quat))