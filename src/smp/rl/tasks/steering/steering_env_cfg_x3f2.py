"""X3_F2 steering task with SMP guidance.
每个环境会采样一个目标 xy 速度方向和一个目标朝向方向。
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from smp.rl.env_cfg_x3f2 import x3f2_smp_env_cfg
from smp.rl.rewards import task_smp_product
from smp.rl.tasks.steering import mdp


def x3f2_steering_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the X3_F2 steering env cfg with SMP guidance."""
  cfg = x3f2_smp_env_cfg(play=play)

  # --- Commands ------------------------------------------------------------
  cfg.commands["steering"] = mdp.SteeringCommandCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),
    rand_tar_dir=True,
    rand_face_dir=True,
    tar_speed_min=0.5,
    tar_speed_max=2.0,
    debug_vis=True,
  )

  # --- Observations --------------------------------------------------------
  command_obs = ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": "steering"},
  )
  cfg.observations["actor"].terms["command"] = command_obs
  cfg.observations["critic"].terms["command"] = command_obs

  # --- Rewards -------------------------------------------------------------
  # task = 0.5 * 速度跟踪 + 0.5 * 朝向对齐
  # 最后由 task_smp_product 和 SMP reward 相乘。
  cfg.rewards["task_smp_product"] = RewardTermCfg(
    func=task_smp_product,
    weight=1.0,
    params={
      "task_terms": (
        (
          mdp.steering_target_velocity,
          0.5,
          {"command_name": "steering", "vel_err_scale": 1.0},
        ),
        (
          mdp.steering_face_direction,
          0.5,
          {"command_name": "steering"},
        ),
      ),
    },
  )

  # --- Events --------------------------------------------------------------
  cfg.events["init_smp_state"].params["ckpt_path"] = (
    "logs/pretrain/pretrain_x3f2_steering/pretrained.pt"
  )

  # --- Terminations --------------------------------------------------------
  cfg.terminations["base_too_low"] = TerminationTermCfg(
    func=mdp.root_height_below_minimum,
    params={
      "minimum_height": 0.35,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  return cfg