"""将 X3_F2 的 CSV 动作文件转换成窗口化 NPZ 数据集。

每个输出 NPZ 文件包含一个 ``windows`` 数组，形状为：
  (N, window_size, F)

当前 X3_F2 使用 14 个关节和 3 个末端点，单帧 feature 维度为 38：

  root_pos        (3)              xy 位于最后一帧 heading-inv 坐标系，
                                    z 保持世界坐标
  root_rot        (6)              heading_inv(T) ⊗ root_quat[t] 的 6D tan-norm 表示
  joint_pos       (num_joints=14)  原始关节角
  ee_pos          (num_ee*3=9)     末端点相对当前 root 的位置，
                                    再旋转到最后一帧 heading-inv 坐标系
  root_lin_vel    (3)              最后一帧 heading-inv 坐标系下的根线速度
  root_ang_vel    (3)              最后一帧 heading-inv 坐标系下的根角速度

所有空间量都锚定到窗口最后一帧的 yaw-only 局部坐标系：
  原点 = 最后一帧 pelvis 位置
  x 轴方向 = 最后一帧 pelvis 的 heading 方向

使用方法：
  uv run scripts/csv_to_npz_x3f2.py --input-dir datasets/csv_x3f2 --output-dir datasets/npz_x3f2
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.scripts.csv_to_npz import MotionLoader as CsvMotionLoader
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  quat_conjugate,
  quat_mul,
  yaw_quat,
)

from smp.rl.env_cfg_x3f2 import x3f2_smp_env_cfg
from smp.utils import detect_device

# CSV 每帧格式应为：
#   root_pos(3) + root_quat(4) + 下面 14 个关节角
# 关节顺序必须和这里完全一致。
JOINT_NAMES: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
)

NUM_JOINTS = len(JOINT_NAMES)

# X3_F2 的末端点。
# 这里选择：左脚、右脚、躯干。
# 这个顺序后面需要和在线 feature buffer / feature_to_state.py 保持一致。
EE_BODY_NAMES: tuple[str, ...] = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "torso_link",
)

NUM_EE = len(EE_BODY_NAMES)


@dataclass
class Cfg:
  input_dir: str = "datasets/csv_x3f2"
  """输入 CSV 动作文件所在目录。"""

  output_dir: str = "datasets/npz_x3f2"
  """输出 NPZ 文件保存目录。"""

  window_size: int = 10
  """每个窗口包含的帧数。"""

  stride: int = 1
  """滑动窗口步长。"""

  input_fps: int = 30
  """输入 CSV 的帧率。"""

  output_fps: int = 50
  """插值后的输出帧率，同时也作为仿真帧率使用。"""

  device: str = ""
  """计算设备。空字符串表示自动选择 cuda 或 cpu。"""

  shard_index: int = 0
  """当前分片编号，用于并行处理数据集。"""

  num_shards: int = 1
  """总分片数量，用于并行处理数据集。"""


def _setup_sim(device: str) -> tuple[Simulation, Scene]:
  """构建 X3_F2 仿真环境，用于前向运动学计算。"""
  sim_cfg = SimulationCfg()
  env_cfg = x3f2_smp_env_cfg()

  scene = Scene(env_cfg.scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  return sim, scene


@torch.no_grad()
def _fk_motion(
  csv_path: Path,
  sim: Simulation,
  scene: Scene,
  joint_indexes: torch.Tensor,
  ee_indexes: torch.Tensor,
  input_fps: int,
  output_fps: int,
) -> tuple[
  torch.Tensor,  # base_pos
  torch.Tensor,  # base_quat
  torch.Tensor,  # base_lin_vel
  torch.Tensor,  # base_ang_vel
  torch.Tensor,  # ee_pos，形状为 (T, num_ee, 3)
  torch.Tensor,  # joint_pos，形状为 (T, num_joints)
  torch.Tensor,  # joint_vel，形状为 (T, num_joints)
]:
  """将 CSV 动作逐帧写入仿真，并通过前向运动学计算末端点位置。"""
  motion = CsvMotionLoader(
    motion_file=str(csv_path),
    input_fps=input_fps,
    output_fps=output_fps,
    device=sim.device,
  )
  robot: Entity = scene["robot"]

  ee_pos_list: list[torch.Tensor] = []

  scene.reset()
  for _ in range(motion.output_frames):
    state, _ = motion.get_next_state()
    base_pos, base_rot, base_lin_vel, base_ang_vel, dof_pos, dof_vel = state

    # 写入根节点状态。
    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = base_pos
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = base_rot
    root_states[:, 7:10] = base_lin_vel
    root_states[:, 10:] = base_ang_vel
    robot.write_root_state_to_sim(root_states)

    # 写入 14 个关节的位置和速度。
    joint_pos_full = robot.data.default_joint_pos.clone()
    joint_vel_full = robot.data.default_joint_vel.clone()
    joint_pos_full[:, joint_indexes] = dof_pos
    joint_vel_full[:, joint_indexes] = dof_vel
    robot.write_joint_state_to_sim(joint_pos_full, joint_vel_full)

    # 前向运动学，读取左脚、右脚和 torso_link 的世界坐标。
    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    ee_pos_list.append(robot.data.body_link_pos_w[0, ee_indexes].clone())

  return (
    motion.motion_base_poss,
    motion.motion_base_rots,
    motion.motion_base_lin_vels,
    motion.motion_base_ang_vels,
    torch.stack(ee_pos_list),
    motion.motion_dof_poss,
    motion.motion_dof_vels,
  )


def _tan_norm_from_quat(quat: torch.Tensor) -> torch.Tensor:
  """将 wxyz 四元数转换为 6D tan-norm 表示。

  这里使用旋转矩阵的第 0 列和第 2 列：
    [x_axis, z_axis]

  输入形状为 ``(..., 4)``，输出形状为 ``(..., 6)``。
  """
  mat = matrix_from_quat(quat)
  col0 = mat[..., :, 0]
  col2 = mat[..., :, 2]
  return torch.cat([col0, col2], dim=-1)


def _compute_windows(
  base_pos: torch.Tensor,
  base_quat: torch.Tensor,
  base_lin_vel: torch.Tensor,
  base_ang_vel: torch.Tensor,
  ee_pos: torch.Tensor,
  joint_pos: torch.Tensor,
  window_size: int,
  stride: int,
) -> torch.Tensor | None:
  """切分窗口，并计算每一帧的 motion feature。

  所有空间量都锚定到窗口最后一帧的 yaw-only 局部坐标系。
  关节速度不写入最终 feature。

  返回形状：
    (num_windows, window_size, 3 + 6 + J + E*3 + 3 + 3)

  当前 X3_F2：
    J = 14
    E = 3
    feature_dim = 38
  """
  T = base_pos.shape[0]
  if T < window_size:
    return None

  E = ee_pos.shape[1]
  J = joint_pos.shape[1]
  starts = torch.arange(
    0, T - window_size + 1, stride, device=base_pos.device, dtype=torch.long
  )
  offsets = torch.arange(window_size, device=base_pos.device, dtype=torch.long)
  win_idx = starts[:, None] + offsets[None, :]
  N, W = win_idx.shape[0], window_size

  flat_idx = win_idx.reshape(-1)
  win_base_pos = base_pos.index_select(0, flat_idx).reshape(N, W, 3)
  win_base_quat = base_quat.index_select(0, flat_idx).reshape(N, W, 4)
  win_base_lin_vel = base_lin_vel.index_select(0, flat_idx).reshape(N, W, 3)
  win_base_ang_vel = base_ang_vel.index_select(0, flat_idx).reshape(N, W, 3)
  win_ee_pos = ee_pos.index_select(0, flat_idx).reshape(N, W, E, 3)
  win_joint = joint_pos.index_select(0, flat_idx).reshape(N, W, J)

  anchor_pos_T = win_base_pos[:, -1, :]
  anchor_quat_T = win_base_quat[:, -1, :]
  yaw_T = yaw_quat(anchor_quat_T)

  heading_inv_T_WF = quat_conjugate(yaw_T)[:, None, :].expand(N, W, 4).reshape(-1, 4)
  yaw_T_W = yaw_T[:, None, :].expand(N, W, 4).reshape(-1, 4)

  # root_pos：xy 转到最后一帧 heading-inv 坐标系，z 保持世界高度。
  root_offset = win_base_pos - anchor_pos_T[:, None, :]
  root_pos_local = quat_apply_inverse(yaw_T_W, root_offset.reshape(-1, 3)).reshape(
    N, W, 3
  )
  root_pos_local = root_pos_local.clone()
  root_pos_local[..., 2] = win_base_pos[..., 2]

  # root_rot：heading_inv(T) ⊗ root_quat[t]，再转成 6D 表示。
  root_rot_local_quat = quat_mul(
    heading_inv_T_WF, win_base_quat.reshape(-1, 4)
  ).reshape(N, W, 4)
  root_rot_6d = _tan_norm_from_quat(root_rot_local_quat)

  # ee_pos：末端点相对当前 root 的偏移，再转到最后一帧 heading-inv 坐标系。
  ee_offset_w = win_ee_pos - win_base_pos[:, :, None, :]
  yaw_T_E = yaw_T[:, None, None, :].expand(N, W, E, 4).reshape(-1, 4)
  ee_pos_local = quat_apply_inverse(yaw_T_E, ee_offset_w.reshape(-1, 3)).reshape(
    N, W, E * 3
  )

  # 根节点线速度和角速度也转到最后一帧 heading-inv 坐标系。
  lin_vel_local = quat_apply_inverse(yaw_T_W, win_base_lin_vel.reshape(-1, 3)).reshape(
    N, W, 3
  )
  ang_vel_local = quat_apply_inverse(yaw_T_W, win_base_ang_vel.reshape(-1, 3)).reshape(
    N, W, 3
  )

  return torch.cat(
    [
      root_pos_local,
      root_rot_6d,
      win_joint,
      ee_pos_local,
      lin_vel_local,
      ang_vel_local,
    ],
    dim=-1,
  )


def main(cfg: Cfg) -> None:
  if not cfg.device:
    cfg.device = detect_device()
  print(f"Device: {cfg.device}")

  in_dir = Path(cfg.input_dir)
  out_dir = Path(cfg.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  csv_files = sorted(in_dir.glob("*.csv"))
  if not csv_files:
    msg = f"No CSV files found in {in_dir}"
    raise FileNotFoundError(msg)

  if cfg.num_shards > 1:
    csv_files = csv_files[cfg.shard_index :: cfg.num_shards]
    print(f"Shard {cfg.shard_index}/{cfg.num_shards}: {len(csv_files)} files")

  sim, scene = _setup_sim(cfg.device)
  robot: Entity = scene["robot"]

  # 按关节名查找仿真内部 joint index。
  joint_indexes = torch.tensor(
    robot.find_joints(list(JOINT_NAMES), preserve_order=True)[0],
    dtype=torch.long,
    device=sim.device,
  )

  # 按 body 名查找末端点 body index。
  ee_indexes = torch.tensor(
    robot.find_bodies(list(EE_BODY_NAMES), preserve_order=True)[0],
    dtype=torch.long,
    device=sim.device,
  )

  feature_dims = [3, 6, NUM_JOINTS, NUM_EE * 3, 3, 3]
  total_feature_dim = sum(feature_dims)

  print(f"Files: {len(csv_files)} in {in_dir}")
  print(f"Output: {out_dir}")
  print(f"Window: size={cfg.window_size} stride={cfg.stride} fps={cfg.output_fps}")
  print(f"End-effectors: {NUM_EE} {EE_BODY_NAMES} | Joints: {NUM_JOINTS}")
  print(
    f"Feature dim: {total_feature_dim} "
    f"(= 3 root_pos + 6 root_rot + {NUM_JOINTS} joint_pos + {NUM_EE * 3} "
    f"ee_pos + 3 lin_vel + 3 ang_vel)"
  )

  for i, csv_path in enumerate(csv_files):
    print(f"\n[{i + 1}/{len(csv_files)}] {csv_path.name}")
    (
      base_pos,
      base_quat,
      base_lin_vel,
      base_ang_vel,
      ee_pos,
      joint_pos,
      joint_vel,
    ) = _fk_motion(
      csv_path,
      sim,
      scene,
      joint_indexes,
      ee_indexes,
      input_fps=cfg.input_fps,
      output_fps=cfg.output_fps,
    )

    if joint_pos.shape[-1] != NUM_JOINTS:
      msg = (
        f"{csv_path.name}: expected {NUM_JOINTS} dof columns, got {joint_pos.shape[-1]}"
      )
      raise ValueError(msg)

    # 当前 feature 不使用 joint_vel。
    del joint_vel

    windows = _compute_windows(
      base_pos,
      base_quat,
      base_lin_vel,
      base_ang_vel,
      ee_pos,
      joint_pos,
      cfg.window_size,
      cfg.stride,
    )
    if windows is None:
      print(f"  [SKIP] too short for window_size={cfg.window_size}")
      continue

    out_path = out_dir / f"{csv_path.stem}.npz"
    np.savez_compressed(
      out_path,
      windows=windows.cpu().numpy().astype(np.float32),
      fps=np.array([cfg.output_fps], dtype=np.float32),
      window_size=np.array([cfg.window_size], dtype=np.int32),
      stride=np.array([cfg.stride], dtype=np.int32),
      ee_body_names=np.array(EE_BODY_NAMES),
      feature_dims=np.array(feature_dims, dtype=np.int32),
    )
    print(f"  saved {out_path.name}: windows={tuple(windows.shape)}")


if __name__ == "__main__":
  main(tyro.cli(Cfg))