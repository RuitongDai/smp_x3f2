"""X3_F2 forward task with SMP guidance.
任务目标：机器人始终朝世界系 +x 方向前进，只随机目标速度。
奖励结构：
  task = velocity tracking
  final reward = task × SMP guidance
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


def x3f2_forward_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the X3_F2 forward env cfg with SMP guidance."""
  cfg = x3f2_smp_env_cfg(play=play)

  # --- Commands ------------------------------------------------------------
  # rand_tar_dir=False 表示目标速度方向固定为 +x。
  # rand_face_dir=False 表示目标朝向也固定为 +x。
  cfg.commands["steering"] = mdp.SteeringCommandCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),
    rand_tar_dir=False,
    rand_face_dir=False,
    tar_speed_min=0.5,
    tar_speed_max=3.0,
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
  # forward 只做目标速度跟踪。
  # task_smp_product 内部会把 task reward 和 SMP guidance reward 相乘。
  cfg.rewards["task_smp_product"] = RewardTermCfg(
    func=task_smp_product,
    weight=1.0,
    params={
      "task_terms": (
        (
          mdp.steering_target_velocity,
          1.0,
          {"command_name": "steering", "vel_err_scale": 0.5},
        ),
      ),
    },
  )

  # --- Events --------------------------------------------------------------
  cfg.events["init_smp_state"].params["ckpt_path"] = (
    "logs/pretrain/pretrain_x3f2_loco/pretrained.pt"
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