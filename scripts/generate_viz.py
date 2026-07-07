"""使用训练好的 X3_F2 SMP diffusion 模型生成 motion window 并可视化。

这个脚本用于检查 diffusion 预训练模型生成的动作质量。

生成流程：
1. 从 checkpoint 加载 denoiser 和 q_low / q_high。
2. 从标准高斯噪声开始做 DDPM 反向采样。
3. 得到一个反归一化后的 motion window。
4. 使用 feature_to_state.py 把 feature window 还原成 pelvis / joint / EE 轨迹。
5. 在 viser viewer 中播放生成结果。

注意：
当前 feature_to_state.py 的 38 维布局：
  root_pos(3) + root_rot(6) + joint_pos(14) + ee_pos(9) + root_lin_vel(3) + root_ang_vel(3)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
import viser
from mjlab.entity import Entity
from mjlab.viewer.viser.scene import MjlabViserScene

from smp.pretrain.model import DiffusionDenoiser
from smp.pretrain.scheduler import DDPMScheduler
from smp.sampling.feature_to_state import (
  NUM_EE,
  window_to_ee_trajectories,
  window_to_pelvis_trajectory,
)
from smp.utils import detect_device


@dataclass
class Cfg:
  ckpt_path: str = ""
  """本地 SMP diffusion checkpoint 路径，通常是 pretrained.pt。"""

  wandb_run: str = ""
  """W&B run 路径，格式为 '<entity>/<project>/<run_id>'。和 ckpt_path 二选一。"""

  device: str = ""
  """计算设备。空字符串表示自动选择。"""

  fps: float = 50.0
  """播放帧率。"""


def _resolve_ckpt_path(cfg: Cfg) -> str:
  """解析 checkpoint 路径。

  如果传入 ckpt_path，就直接使用本地文件。
  如果传入 wandb_run，就从 W&B 下载对应的 .pt 文件。
  """
  if bool(cfg.ckpt_path) == bool(cfg.wandb_run):
    msg = "必须且只能指定 --ckpt-path 或 --wandb-run 其中一个"
    raise ValueError(msg)

  if cfg.ckpt_path:
    return cfg.ckpt_path

  import wandb

  api = wandb.Api()
  run = api.run(cfg.wandb_run)
  pt_files = [f for f in run.files() if f.name.endswith(".pt")]

  if not pt_files:
    msg = f"W&B run 中没有 .pt 文件: {cfg.wandb_run}"
    raise FileNotFoundError(msg)

  target = next(
    (f for f in pt_files if Path(f.name).name == "pretrained.pt"),
    sorted(pt_files, key=lambda f: f.name)[-1],
  )

  download_dir = Path("logs") / "wandb_ckpt_cache" / cfg.wandb_run.replace("/", "_")
  download_dir.mkdir(parents=True, exist_ok=True)

  target.download(root=str(download_dir), replace=True)
  local = download_dir / target.name

  print(f"Downloaded {target.name} from {cfg.wandb_run} -> {local}")

  return str(local)


def _build_model_and_scheduler(
  ckpt: dict,
  device: torch.device,
) -> tuple[DiffusionDenoiser, DDPMScheduler, np.ndarray, np.ndarray]:
  """从 checkpoint 构建 denoiser 和 DDPM scheduler。"""
  cfg = ckpt["cfg"]

  model = DiffusionDenoiser(
    feature_dim=cfg["feature_dim"],
    window_size=cfg["window_size"],
    d_model=cfg.get("d_model", 256),
    nhead=cfg.get("nhead", 4),
    num_layers=cfg.get("num_layers", 2),
    dropout=cfg.get("dropout", 0.0),
  ).to(device)

  state = ckpt.get("model_ema") or ckpt["model"]
  model.load_state_dict(state)
  model.eval()

  scheduler = DDPMScheduler(
    num_timesteps=cfg.get("num_timesteps", 50),
  ).to(device)

  return model, scheduler, ckpt["q_low"], ckpt["q_high"]


def _setup_x3f2_sim(device: str):
  """构建单个 X3_F2 仿真环境，用于 viser 可视化。"""
  from mjlab.scene import Scene
  from mjlab.sim.sim import Simulation, SimulationCfg

  from smp.rl.env_cfg_x3f2 import x3f2_smp_env_cfg

  sim_cfg = SimulationCfg()
  env_cfg = x3f2_smp_env_cfg(play=True)

  scene = Scene(env_cfg.scene, device=device)
  model = scene.compile()

  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  return sim, scene


def _quantile_denormalize(
  x: torch.Tensor,
  q_low: torch.Tensor,
  q_high: torch.Tensor,
) -> torch.Tensor:
  """把 [-1, 1] 区间的 feature 反归一化回真实数值范围。"""
  return (x + 1.0) / 2.0 * (q_high - q_low) + q_low


@torch.no_grad()
def _run_generate(
  model: DiffusionDenoiser,
  scheduler: DDPMScheduler,
  q_low: np.ndarray,
  q_high: np.ndarray,
  window_size: int,
  feature_dim: int,
  device: torch.device,
) -> torch.Tensor:
  """执行一次无条件 DDPM 采样，返回形状为 (W, F) 的反归一化 window。"""
  x_t = torch.randn(1, window_size, feature_dim, device=device)

  for t in reversed(range(scheduler.num_timesteps)):
    t_batch = torch.full((1,), t, dtype=torch.long, device=device)
    eps = model(x_t, t_batch)
    x_t = scheduler.step(eps, x_t, t)

  q_low_t = torch.from_numpy(q_low).float().to(device)
  q_high_t = torch.from_numpy(q_high).float().to(device)

  return _quantile_denormalize(x_t.squeeze(0), q_low_t, q_high_t).cpu()


def _write_pose_to_robot(
  robot: Entity,
  pelvis_pos: np.ndarray,
  pelvis_quat_wxyz: np.ndarray,
  joint_pos: np.ndarray,
  device: str,
) -> None:
  """把一帧生成结果写入机器人状态。"""
  root = robot.data.default_root_state.clone()
  root[:, 0:3] = torch.as_tensor(pelvis_pos, device=device, dtype=root.dtype)
  root[:, 3:7] = torch.as_tensor(pelvis_quat_wxyz, device=device, dtype=root.dtype)
  robot.write_root_state_to_sim(root)

  jp = robot.data.default_joint_pos.clone()
  jp[:] = torch.as_tensor(joint_pos, device=device, dtype=jp.dtype)

  jv = robot.data.default_joint_vel.clone()
  robot.write_joint_state_to_sim(jp, jv)


def main(cfg: Cfg) -> None:
  device_str = cfg.device or detect_device()
  device = torch.device(device_str)

  print(f"Device: {device_str}")

  ckpt_path = _resolve_ckpt_path(cfg)
  ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

  model, scheduler, q_low, q_high = _build_model_and_scheduler(ckpt, device)

  print(f"Loaded checkpoint epoch={ckpt.get('epoch')} from {ckpt_path}")

  feature_dim = int(ckpt["cfg"]["feature_dim"])
  window_size = int(ckpt["cfg"]["window_size"])

  if feature_dim != 38:
    print(f"WARNING: 当前 checkpoint feature_dim={feature_dim}，X3_F2 当前期望是 38。")

  sim_device = device_str
  sim, scene = _setup_x3f2_sim(sim_device)

  robot: Entity = scene["robot"]
  mj_model = sim.mj_model

  # 将生成 window 的最后一帧放在机器人默认站立位置。
  anchor_pelvis_pos = robot.data.default_root_state[0, 0:3].detach().cpu()
  anchor_pelvis_quat = robot.data.default_root_state[0, 3:7].detach().cpu()

  def run() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """重新采样一次 motion window，并还原成可播放的轨迹。"""
    pred_denorm = _run_generate(
      model,
      scheduler,
      q_low,
      q_high,
      window_size,
      feature_dim,
      device,
    )

    p_pos, p_quat, p_joint = window_to_pelvis_trajectory(
      pred_denorm,
      anchor_pelvis_pos,
      anchor_pelvis_quat,
    )

    ee_pos = window_to_ee_trajectories(pred_denorm, p_pos, p_quat)

    return (
      p_pos.cpu().numpy(),
      p_quat.cpu().numpy(),
      p_joint.cpu().numpy(),
      ee_pos.cpu().numpy(),
    )

  state: dict = {"pred": run()}

  server = viser.ViserServer()
  viser_scene = MjlabViserScene(server, mj_model, num_envs=1)
  viser_scene.debug_visualization_enabled = True

  # 显示生成 feature 中的末端点位置。
  ee_points = server.scene.add_point_cloud(
    name="/fixed_bodies/predicted_ee_positions",
    points=np.zeros((NUM_EE, 3), dtype=np.float32),
    colors=np.tile(np.array([255, 80, 0], dtype=np.uint8), (NUM_EE, 1)),
    point_size=0.03,
  )

  with server.gui.add_folder("Generate"):
    frame_slider = server.gui.add_slider(
      "Frame",
      min=0,
      max=window_size - 1,
      step=1,
      initial_value=0,
    )
    play_btn = server.gui.add_button("Play / Pause")
    resample_btn = server.gui.add_button("Resample")

  playing = {"v": True}

  @play_btn.on_click
  def _(_evt) -> None:
    playing["v"] = not playing["v"]

  @resample_btn.on_click
  def _(_evt) -> None:
    state["pred"] = run()

  def render(frame: int) -> None:
    """渲染指定帧。"""
    p_pos, p_quat, p_joint, ee_pos = state["pred"]

    _write_pose_to_robot(
      robot,
      p_pos[frame],
      p_quat[frame],
      p_joint[frame],
      sim_device,
    )

    sim.forward()

    wd = sim.wp_data
    viser_scene.update_from_arrays(
      body_xpos=np.asarray(wd.xpos.numpy()),
      body_xmat=np.asarray(wd.xmat.numpy()),
      qpos=np.asarray(wd.qpos.numpy()),
      env_idx=0,
    )

    ee_points.points = ee_pos[frame]
    viser_scene.refresh_visualization()

  print("Viser server running. Open the printed URL.")

  dt_play = 1.0 / cfg.fps

  try:
    while True:
      render(int(frame_slider.value))

      if playing["v"]:
        nxt = (int(frame_slider.value) + 1) % window_size
        frame_slider.value = nxt

      time.sleep(dt_play)

  except KeyboardInterrupt:
    print("Shutting down.")


if __name__ == "__main__":
  main(tyro.cli(Cfg))