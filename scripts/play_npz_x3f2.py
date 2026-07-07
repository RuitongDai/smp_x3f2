"""播放处理好的 X3_F2 NPZ 窗口数据。

这个脚本用于检查 csv_to_npz_x3f2.py 生成的 NPZ 是否合理。

当前假设单帧 feature 维度为 38：
  root_pos      3
  root_rot_6d   6
  joint_pos     14
  ee_pos        9   # 左脚、右脚、torso_link
  root_lin_vel  3
  root_ang_vel  3

注意：
  NPZ 里的 root_pos / root_rot 已经是窗口局部坐标系下的数据，
  不是原始全局轨迹，所以这里只适合检查单个 window 内的姿态和关节顺序。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
  sys.path.insert(0, str(SRC_ROOT))

from x3.x3_constants import get_spec

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

# 当前 feature 分段。
ROOT_POS_DIM = 3
ROOT_ROT_DIM = 6
EE_POS_DIM = 9
ROOT_LIN_VEL_DIM = 3
ROOT_ANG_VEL_DIM = 3
FEATURE_DIM = ROOT_POS_DIM + ROOT_ROT_DIM + NUM_JOINTS + EE_POS_DIM + ROOT_LIN_VEL_DIM + ROOT_ANG_VEL_DIM


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
  """归一化向量，避免除零。"""
  norm = np.linalg.norm(v)
  if norm < eps:
    return v
  return v / norm


def _quat_from_matrix(mat: np.ndarray) -> np.ndarray:
  """把 3x3 旋转矩阵转换成 MuJoCo 使用的 wxyz 四元数。"""
  trace = np.trace(mat)

  if trace > 0.0:
    s = np.sqrt(trace + 1.0) * 2.0
    qw = 0.25 * s
    qx = (mat[2, 1] - mat[1, 2]) / s
    qy = (mat[0, 2] - mat[2, 0]) / s
    qz = (mat[1, 0] - mat[0, 1]) / s
  elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
    s = np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2.0
    qw = (mat[2, 1] - mat[1, 2]) / s
    qx = 0.25 * s
    qy = (mat[0, 1] + mat[1, 0]) / s
    qz = (mat[0, 2] + mat[2, 0]) / s
  elif mat[1, 1] > mat[2, 2]:
    s = np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2.0
    qw = (mat[0, 2] - mat[2, 0]) / s
    qx = (mat[0, 1] + mat[1, 0]) / s
    qy = 0.25 * s
    qz = (mat[1, 2] + mat[2, 1]) / s
  else:
    s = np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2.0
    qw = (mat[1, 0] - mat[0, 1]) / s
    qx = (mat[0, 2] + mat[2, 0]) / s
    qy = (mat[1, 2] + mat[2, 1]) / s
    qz = 0.25 * s

  quat = np.array([qw, qx, qy, qz], dtype=np.float64)
  return _normalize(quat)


def _quat_from_tan_norm(rot_6d: np.ndarray) -> np.ndarray:
  """把 tan-norm 6D 旋转表示还原成 wxyz 四元数。

  csv_to_npz_x3f2.py 里保存的是旋转矩阵的第 0 列和第 2 列：
    rot_6d = [x_axis, z_axis]

  这里重新构造右手坐标系：
    y_axis = z_axis × x_axis
    z_axis = x_axis × y_axis
  """
  x_axis = _normalize(rot_6d[0:3])
  z_axis = rot_6d[3:6]

  # 去掉 z 在 x 方向上的投影，减少数值误差。
  z_axis = z_axis - np.dot(z_axis, x_axis) * x_axis
  z_axis = _normalize(z_axis)

  y_axis = _normalize(np.cross(z_axis, x_axis))
  z_axis = _normalize(np.cross(x_axis, y_axis))

  rot_mat = np.stack([x_axis, y_axis, z_axis], axis=1)
  return _quat_from_matrix(rot_mat)


def _resolve_npz_path(npz_path: str) -> Path:
  """支持传入单个 npz 文件，也支持传入文件夹。"""
  path = Path(npz_path)

  if path.is_file():
    return path

  if path.is_dir():
    files = sorted(path.glob("*.npz"))
    if not files:
      raise FileNotFoundError(f"目录里没有 npz 文件: {path}")
    print(f"传入的是目录，默认播放第一个文件: {files[0]}")
    return files[0]

  raise FileNotFoundError(f"找不到 npz 文件或目录: {path}")


def _load_windows(npz_path: Path) -> np.ndarray:
  """读取 NPZ 里的 windows 数据。"""
  data = np.load(npz_path, allow_pickle=True)

  if "windows" not in data:
    raise KeyError(f"{npz_path} 里没有 windows 字段")

  windows = data["windows"]

  if windows.ndim != 3:
    raise ValueError(f"windows 应该是 3 维，实际形状为: {windows.shape}")

  if windows.shape[-1] != FEATURE_DIM:
    raise ValueError(
      f"feature 维度不对，期望 {FEATURE_DIM}，实际 {windows.shape[-1]}。"
      "如果你改过末端点数量，需要同步改这个播放脚本的 FEATURE_DIM。"
    )

  print(f"NPZ 文件: {npz_path}")
  print(f"windows 形状: {windows.shape}")

  if "feature_dims" in data:
    print(f"feature_dims: {data['feature_dims']}")

  if "ee_body_names" in data:
    print(f"ee_body_names: {data['ee_body_names']}")

  return windows


def _set_robot_state(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  frame: np.ndarray,
) -> None:
  """把一帧 feature 写入 MuJoCo qpos。"""
  root_pos = frame[0:3]
  root_rot_6d = frame[3:9]
  joint_pos = frame[9 : 9 + NUM_JOINTS]

  root_quat = _quat_from_tan_norm(root_rot_6d)

  # 写入 floating base。
  free_joint_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "floating_base_joint",
  )
  if free_joint_id < 0:
    raise ValueError("模型里找不到 floating_base_joint")

  free_qpos_adr = model.jnt_qposadr[free_joint_id]
  data.qpos[free_qpos_adr : free_qpos_adr + 3] = root_pos
  data.qpos[free_qpos_adr + 3 : free_qpos_adr + 7] = root_quat

  # 写入 14 个关节。
  for name, value in zip(JOINT_NAMES, joint_pos):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
      raise ValueError(f"模型里找不到关节: {name}")

    qpos_adr = model.jnt_qposadr[joint_id]
    data.qpos[qpos_adr] = value

  # 这里是按数据集强制摆姿态，不做动力学仿真，所以速度清零即可。
  data.qvel[:] = 0.0
  mujoco.mj_forward(model, data)


def _play_window(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  viewer: mujoco.viewer.Handle,
  window: np.ndarray,
  fps: float,
  loop: bool,
) -> None:
  """播放单个 window。"""
  frame_dt = 1.0 / fps

  while viewer.is_running():
    for frame in window:
      if not viewer.is_running():
        return

      start_time = time.time()

      _set_robot_state(model, data, frame)
      viewer.sync()

      sleep_time = frame_dt - (time.time() - start_time)
      if sleep_time > 0.0:
        time.sleep(sleep_time)

    if not loop:
      return


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--npz-path",
    type=str,
    required=True,
    help="要播放的 npz 文件路径，也可以传入 npz 文件夹。",
  )
  parser.add_argument(
    "--window-index",
    type=int,
    default=0,
    help="播放第几个 window。设置为 -1 表示随机选择一个 window。",
  )
  parser.add_argument(
    "--fps",
    type=float,
    default=20.0,
    help="播放帧率。窗口只有 10 帧时，可以设置小一点方便观察。",
  )
  parser.add_argument(
    "--no-loop",
    action="store_true",
    help="只播放一次，不循环。",
  )
  args = parser.parse_args()

  npz_path = _resolve_npz_path(args.npz_path)
  windows = _load_windows(npz_path)

  num_windows = windows.shape[0]
  if args.window_index < 0:
    window_index = int(np.random.randint(0, num_windows))
  else:
    window_index = args.window_index

  if window_index >= num_windows:
    raise IndexError(f"window_index 超出范围: {window_index} >= {num_windows}")

  print(f"播放 window_index: {window_index}")
  print("提示：这个脚本是按窗口局部坐标播放，用来检查姿态和关节顺序。")

  spec = get_spec()
  model = spec.compile()
  data = mujoco.MjData(model)

  window = windows[window_index].astype(np.float64)

  with mujoco.viewer.launch_passive(model, data) as viewer:
    _play_window(
      model=model,
      data=data,
      viewer=viewer,
      window=window,
      fps=args.fps,
      loop=not args.no_loop,
    )


if __name__ == "__main__":
  main()