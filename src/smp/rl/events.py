"""SMP RL 的 startup / reset / step 事件。

这些函数由 mjlab 的 event manager 调用。
作用是把 diffusion prior 接到普通的 ManagerBasedRlEnv 上。

核心流程：
1. startup 时加载预训练 denoiser。
2. startup 时预先采样一批 GSI window。
3. reset 时从 GSI pool 里随机取一个 window 初始化机器人状态。
4. step 时定期刷新 GSI pool。
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply, quat_mul, yaw_quat

from smp.rl.utils import DiffNormalizer, MotionFeatureBuffer, load_denoiser
from smp.sampling.feature_to_state import (
  EE_BODY_NAMES,
  NUM_EE,
  rot6d_to_quat,
  slice_features,
)

# X3_F2 当前使用 14 个关节。
NUM_JOINTS = 14


def _maybe_compile(model, compile_model: bool, compile_mode: str | None):
  """按需对 denoiser 执行 torch.compile。"""
  if not compile_model:
    return model

  torch.set_float32_matmul_precision("high")

  try:
    import torch._inductor.config as _ic

    # 关闭 shape padding，避免部分 Inductor / TF32 组合下的 pad_mm 问题。
    _ic.shape_padding = False
  except ImportError:
    pass

  if compile_mode is not None:
    return torch.compile(model, fullgraph=True, mode=compile_mode)

  return torch.compile(model, fullgraph=True)


def init_smp_state(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  ckpt_path: str = "",
  gsi_buffer_size: int = 4096,
  gsi_batch_size: int = 256,
  compile_model: bool = True,
  compile_mode: str | None = None,
) -> None:
  """startup 事件：初始化 SMP 相关状态。

  具体做的事：
  1. 加载已经预训练好的 diffusion denoiser。
  2. 创建 MotionFeatureBuffer。
  3. 创建 DiffNormalizer。
  4. 预先用 DDPM 采样一批 window，放进 GSI pool。
  5. 调用一次 gsi_reset，把机器人初始化到生成动作的最后一帧。
  """
  del env_ids

  if not ckpt_path:
    msg = (
      "init_smp_state called without `ckpt_path`. "
      "请在 EventTermCfg 里设置 ckpt_path。"
    )
    raise RuntimeError(msg)

  model, scheduler, q_low, q_high, feature_dim, window_size = load_denoiser(
    ckpt_path, env.device
  )

  expected_feature_dim = 3 + 6 + NUM_JOINTS + NUM_EE * 3 + 3 + 3
  if feature_dim != expected_feature_dim:
    msg = (
      f"SMP checkpoint feature_dim 不匹配："
      f"ckpt={feature_dim}, expected={expected_feature_dim}. "
      "请确认预训练模型是用 X3_F2 的 38 维 NPZ 训练出来的。"
    )
    raise ValueError(msg)

  model = _maybe_compile(model, compile_model, compile_mode)

  env._smp_bundle = (  # type: ignore[attr-defined]
    model,
    scheduler,
    q_low,
    q_high,
    feature_dim,
    window_size,
  )

  robot = env.scene["robot"]

  # 找到 feature 里使用的末端点 body index。
  env._smp_ee_indexes = torch.tensor(  # type: ignore[attr-defined]
    robot.find_bodies(list(EE_BODY_NAMES), preserve_order=True)[0],
    dtype=torch.long,
    device=env.device,
  )

  # 这个 buffer 会持续保存最近 window_size 帧的仿真运动状态。
  env._smp_buffer = MotionFeatureBuffer(  # type: ignore[attr-defined]
    num_envs=env.num_envs,
    window_size=window_size,
    num_joints=NUM_JOINTS,
    num_ee=NUM_EE,
    device=env.device,
  )

  env._smp_normalizer = DiffNormalizer(  # type: ignore[attr-defined]
    scheduler.num_timesteps,
    env.device,
  )

  if gsi_buffer_size <= 0:
    msg = f"gsi_buffer_size must be positive, got {gsi_buffer_size}."
    raise ValueError(msg)

  # 预先生成 GSI pool，避免每次 reset 都临时跑完整 DDPM。
  pool_chunks: list[torch.Tensor] = []
  for start in range(0, gsi_buffer_size, gsi_batch_size):
    bsz = min(gsi_batch_size, gsi_buffer_size - start)
    pool_chunks.append(_ddpm_sample(env, bsz))

  env._smp_gsi_pool = torch.cat(pool_chunks, dim=0)  # type: ignore[attr-defined]

  if compile_model and env.num_envs != gsi_batch_size:
    # 预热 reward 路径的 batch shape，避免第一次 step 时才编译卡住。
    with torch.no_grad():
      dummy_x = torch.randn(env.num_envs, window_size, feature_dim, device=env.device)
      dummy_t = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
      _ = model(dummy_x, dummy_t)

  gsi_reset(env)


def _prime_sim_and_buffer(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  window: torch.Tensor,
) -> None:
  """用一个生成的 window 初始化仿真和 SMP feature buffer。

  这里做两件事：
  1. 把 window 最后一帧写入 MuJoCo，作为当前 reset 状态。
  2. 把整个 window 写入 MotionFeatureBuffer，作为 SMP reward 的历史窗口。

  注意：
  buffer 内部保存的是相对 env origin 的运动状态；
  写入仿真时会加上每个 env 的 origin，让多个环境分散在网格上。
  """
  n, W, _ = window.shape
  E = NUM_EE

  parts = slice_features(window)
  root_pos_local = parts["root_pos"]
  root_rot_6d = parts["root_rot"]
  joint_pos = parts["joint_pos"]
  ee_pos_local = parts["ee_pos"].reshape(n, W, E, 3)
  root_lin_vel_local = parts["root_lin_vel"]
  root_ang_vel_local = parts["root_ang_vel"]

  # feature 里没有 joint_vel，这里用相邻帧差分估计。
  control_dt = float(env.cfg.sim.mujoco.timestep) * float(env.cfg.decimation)
  if W > 1:
    joint_vel = torch.zeros_like(joint_pos)
    joint_vel[:, :-1] = (joint_pos[:, 1:] - joint_pos[:, :-1]) / control_dt
    joint_vel[:, -1] = joint_vel[:, -2]
  else:
    joint_vel = torch.zeros_like(joint_pos)

  robot = env.scene["robot"]

  default_root = robot.data.default_root_state[env_ids].clone()
  default_pos = default_root[:, 0:3]
  default_quat = default_root[:, 3:7]

  # 使用默认 root yaw 作为当前环境的 heading。
  yaw_T = yaw_quat(default_quat)
  yaw_T_W = yaw_T[:, None, :].expand(n, W, 4).reshape(-1, 4)

  # 还原 pelvis 世界坐标。
  local_xy = root_pos_local.clone()
  local_xy[..., 2] = 0.0

  world_offset_xy = quat_apply(yaw_T_W, local_xy.reshape(-1, 3)).reshape(n, W, 3)
  pelvis_pos_w = world_offset_xy.clone()
  pelvis_pos_w[..., 0] += default_pos[:, None, 0]
  pelvis_pos_w[..., 1] += default_pos[:, None, 1]
  pelvis_pos_w[..., 2] = root_pos_local[..., 2]

  # 还原 pelvis 姿态。
  root_rot_local_quat = rot6d_to_quat(root_rot_6d.reshape(-1, 6)).reshape(n, W, 4)
  pelvis_quat_w = quat_mul(yaw_T_W, root_rot_local_quat.reshape(-1, 4)).reshape(
    n, W, 4
  )

  # 还原 root 速度。
  lin_vel_w = quat_apply(yaw_T_W, root_lin_vel_local.reshape(-1, 3)).reshape(n, W, 3)
  ang_vel_w = quat_apply(yaw_T_W, root_ang_vel_local.reshape(-1, 3)).reshape(n, W, 3)

  # 还原末端点位置。
  yaw_T_E = yaw_T[:, None, None, :].expand(n, W, E, 4).reshape(-1, 4)
  ee_offset_w = quat_apply(yaw_T_E, ee_pos_local.reshape(-1, 3)).reshape(n, W, E, 3)
  ee_pos_w = ee_offset_w + pelvis_pos_w[:, :, None, :]

  # 写入仿真时加上 env origin。
  origins = env.scene.env_origins[env_ids]
  last_root_state = torch.cat(
    [
      pelvis_pos_w[:, -1] + origins,
      pelvis_quat_w[:, -1],
      lin_vel_w[:, -1],
      ang_vel_w[:, -1],
    ],
    dim=-1,
  )

  robot.write_root_state_to_sim(last_root_state, env_ids=env_ids)
  robot.write_joint_state_to_sim(joint_pos[:, -1], joint_vel[:, -1], env_ids=env_ids)

  # buffer 保持 env-relative，不加 origin。
  buf: MotionFeatureBuffer = env._smp_buffer  # type: ignore[attr-defined]
  buf.reset(
    env_ids,
    pelvis_pos_w,
    pelvis_quat_w,
    lin_vel_w,
    ang_vel_w,
    ee_pos_w,
    joint_pos,
    joint_vel,
  )


@torch.no_grad()
def _ddpm_sample(env: ManagerBasedRlEnv, n: int) -> torch.Tensor:
  """执行 DDPM 反向采样，返回 n 个反归一化后的 motion window。"""
  model, scheduler, q_low, q_high, feature_dim, window_size = env._smp_bundle  # type: ignore[attr-defined]

  x_t = torch.randn(n, window_size, feature_dim, device=env.device)

  for t_int in reversed(range(scheduler.num_timesteps)):
    t = torch.full((n,), t_int, dtype=torch.long, device=env.device)
    eps = model(x_t, t)
    x_t = scheduler.step(eps, x_t, t_int)

  return (x_t + 1.0) / 2.0 * (q_high - q_low) + q_low


@torch.no_grad()
def gsi_refresh(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  num_samples: int = 1024,
  step_interval: int = 2400,
) -> None:
  """step 事件：定期刷新 GSI pool。

  每隔 step_interval 步，重新采样 num_samples 个 window，
  以 FIFO 的方式替换旧的 GSI window。
  """
  del env_ids

  cur = int(env.common_step_counter)
  if cur == 0 or (cur % step_interval) != 0:
    return

  pool: torch.Tensor = env._smp_gsi_pool  # type: ignore[attr-defined]
  pool_size = pool.shape[0]

  if num_samples > pool_size:
    msg = f"num_samples ({num_samples}) cannot exceed pool size ({pool_size})"
    raise ValueError(msg)

  new_windows = _ddpm_sample(env, num_samples)

  head = int(getattr(env, "_smp_gsi_head", 0))
  end = head + num_samples

  if end <= pool_size:
    pool[head:end] = new_windows
  else:
    first = pool_size - head
    pool[head:] = new_windows[:first]
    pool[: end - pool_size] = new_windows[first:]

  env._smp_gsi_head = end % pool_size  # type: ignore[attr-defined]


@torch.no_grad()
def gsi_reset(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None = None) -> None:
  """reset 事件：从 GSI pool 里采样 window 并初始化机器人状态。"""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)

  n = int(env_ids.numel())
  if n == 0:
    return

  pool: torch.Tensor = env._smp_gsi_pool  # type: ignore[attr-defined]
  idx = torch.randint(0, pool.shape[0], (n,), device=env.device)
  window = pool[idx]

  _prime_sim_and_buffer(env, env_ids, window)