"""Unitree G1 机器人常量与物理参数配置。

本文件定义了 G1 的机体资产路径、基于真实电机参数的驱动器模型（包含刚度/阻尼的自动计算）、
初始姿态关键帧以及碰撞检测配置。
"""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.spec_config import CollisionCfg

# ========================================== #
# 1. MJCF 模型与资产路径定义
# ========================================== #
SRC_PATH = Path(__file__).parent.parent
G1_XML: Path = (
  SRC_PATH / "unitree_g1" / "xmls" / "g1.xml"
)
assert G1_XML.exists()

def get_spec() -> mujoco.MjSpec:
  """从 XML 文件加载并返回 MuJoCo 的 MjSpec 对象。"""
  return mujoco.MjSpec.from_file(str(G1_XML))

# ========================================== #
# 2. Actuator (执行器/电机) 物理参数配置
# ========================================== #
# 这里的参数来源于 Unitree 官方的真实电机规格。
# 核心逻辑：利用两级行星齿轮的减速比，计算电机转子到输出轴的“等效反射惯量(Reflected Inertia)”。
# 反射惯量 J_ref = J_rotor * (Gear_Ratio)^2

# --- 5020 电机 (用于手臂等中等扭矩关节) ---
ROTOR_INERTIAS_5020 = (0.139e-4, 0.017e-4, 0.169e-4) # 太阳轮、行星架等各级惯量
GEARS_5020 = (1, 1 + (46 / 18), 1 + (56 / 16))       # 齿轮比
ARMATURE_5020 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_5020, GEARS_5020
)

# --- 7520_14 电机 (用于髋部俯仰/偏航等高扭矩关节) ---
ROTOR_INERTIAS_7520_14 = (0.489e-4, 0.098e-4, 0.533e-4)
GEARS_7520_14 = (1, 4.5, 1 + (48 / 22))
ARMATURE_7520_14 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_7520_14, GEARS_7520_14
)

# --- 7520_22 电机 (用于髋部横滚/膝盖等超高扭矩需求关节) ---
ROTOR_INERTIAS_7520_22 = (0.489e-4, 0.109e-4, 0.738e-4)
GEARS_7520_22 = (1, 4.5, 5)
ARMATURE_7520_22 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_7520_22, GEARS_7520_22
)

# --- 4010 电机 (用于手腕等轻量级关节) ---
ROTOR_INERTIAS_4010 = (0.068e-4, 0.0, 0.0)
GEARS_4010 = (1, 5, 5)
ARMATURE_4010 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_4010, GEARS_4010
)

# --- 构建电机的电气与物理约束模型 ---
ACTUATOR_5020 = ElectricActuator(
  reflected_inertia=ARMATURE_5020, velocity_limit=37.0, effort_limit=25.0
)
ACTUATOR_7520_14 = ElectricActuator(
  reflected_inertia=ARMATURE_7520_14, velocity_limit=32.0, effort_limit=88.0
)
ACTUATOR_7520_22 = ElectricActuator(
  reflected_inertia=ARMATURE_7520_22, velocity_limit=20.0, effort_limit=139.0
)
ACTUATOR_4010 = ElectricActuator(
  reflected_inertia=ARMATURE_4010, velocity_limit=22.0, effort_limit=5.0
)

# ========================================== #
# 3. PD 控制器参数 (基于二阶系统理论自动推导)
# ========================================== #
# 为了保证所有的关节在响应指令时具有相似的动态特性，这里使用了统一的自然频率和阻尼比。
# 公式推导：
# 刚度 Stiffness (Kp) = J_ref * (w_n)^2
# 阻尼 Damping (Kd)   = 2 * zeta * J_ref * w_n

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 目标自然频率 10Hz (转换为 rad/s)
DAMPING_RATIO = 2.0                     # 阻尼比 zeta = 2.0 (过阻尼状态，确保运动平稳无震荡)

# 计算各型号电机的刚度 (Kp)
STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

# 计算各型号电机的阻尼 (Kd)
DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

# --- 将计算好的 PD 参数映射到具体的机器人关节上 ---
G1_ACTUATOR_5020 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_elbow_joint",           # 肘关节
    ".*_shoulder_pitch_joint",  # 肩部俯仰
    ".*_shoulder_roll_joint",   # 肩部横滚
    ".*_shoulder_yaw_joint",    # 肩部偏航
    ".*_wrist_roll_joint",      # 手腕横滚
  ),
  stiffness=STIFFNESS_5020,
  damping=DAMPING_5020,
  effort_limit=ACTUATOR_5020.effort_limit,
  armature=ACTUATOR_5020.reflected_inertia,
)

G1_ACTUATOR_7520_14 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"), # 髋俯仰/髋偏航/腰偏航
  stiffness=STIFFNESS_7520_14,
  damping=DAMPING_7520_14,
  effort_limit=ACTUATOR_7520_14.effort_limit,
  armature=ACTUATOR_7520_14.reflected_inertia,
)

G1_ACTUATOR_7520_22 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"), # 髋横滚/膝盖 (负载最大)
  stiffness=STIFFNESS_7520_22,
  damping=DAMPING_7520_22,
  effort_limit=ACTUATOR_7520_22.effort_limit,
  armature=ACTUATOR_7520_22.reflected_inertia,
)

G1_ACTUATOR_4010 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"), # 手腕俯仰/偏航
  stiffness=STIFFNESS_4010,
  damping=DAMPING_4010,
  effort_limit=ACTUATOR_4010.effort_limit,
  armature=ACTUATOR_4010.reflected_inertia,
)

# --- 特殊并联机构处理 (腰部与脚踝) ---
# 腰部的俯仰/横滚以及脚踝关节实际上是由两个 5020 电机驱动的四连杆(4-bar linkage)并联机构。
# 由于并联机构的等效反射惯量会随构型改变，且精确的几何约束未知，
# 故这里作近似处理：假设标称传动比为 1:1，将标称构型下的关节惯量近似为 2 个电机的惯量之和。
G1_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_pitch_joint", "waist_roll_joint"),
  stiffness=STIFFNESS_5020 * 2,
  damping=DAMPING_5020 * 2,
  effort_limit=ACTUATOR_5020.effort_limit * 2,
  armature=ACTUATOR_5020.reflected_inertia * 2,
)

G1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
  stiffness=STIFFNESS_5020 * 2,
  damping=DAMPING_5020 * 2,
  effort_limit=ACTUATOR_5020.effort_limit * 2,
  armature=ACTUATOR_5020.reflected_inertia * 2,
)


# ========================================== #
# 4. 关键帧 (初始位姿) 配置
# ========================================== #

# 默认站立位姿
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.783675),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.2,
    ".*_elbow_joint": 1.28,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
  },
  joint_vel={".*": 0.0},
)

# 微曲膝准备位姿 (有利于强化学习快速探索出稳定步态)
KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.76),
  joint_pos={
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
  },
  joint_vel={".*": 0.0},
)


# ========================================== #
# 5. 碰撞检测配置
# ========================================== #

# 开启所有碰撞，包括自碰撞。
# 自碰撞的接触维度 condim=1 (仅计算法向力，不计算摩擦力)
# 脚部与地面的碰撞 condim=3 (计算法向力及两个方向的摩擦力)
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1}, # 脚部碰撞优先级更高
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

# 开启碰撞检测，但禁用自碰撞计算 (提高仿真速度)
# contype=0, conaffinity=1 使得机器人各部件之间不会发生碰撞验证
FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

# 极致优化：仅保留脚部的碰撞检测，其他部位如同虚设 (用于极其粗糙的初步训练阶段)
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)


# ========================================== #
# 6. 最终装配与缩放计算
# ========================================== #

# 汇总所有的执行器配置
G1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14,
    G1_ACTUATOR_7520_22,
    G1_ACTUATOR_4010,
    G1_ACTUATOR_WAIST,
    G1_ACTUATOR_ANKLE,
  ),
  soft_joint_pos_limit_factor=0.9, # 软限位系数，防止电机硬撞击机械限位
)

def get_g1_robot_cfg() -> EntityCfg:
  """获取一个全新的 G1 机器人配置实例。

  每次调用都返回一个新实例，防止该配置在多个环境/任务间共享时被意外篡改内部状态。
  """
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=G1_ARTICULATION,
  )

# --- 计算神经网络动作缩放因子 (Action Scale) ---
# 强化学习网络通常输出 [-1, 1] 的标准化动作指令。
# 这里将网络输出映射为实际的关节位置偏移量。
# 公式：Scale = 0.25 * (最大扭矩 / 刚度)。该缩放保证了在网络输出为 1 时，
# PD 控制器产生的弹簧力能达到最大扭矩的 25%，既保证了控制裕度，又防止指令突变导致系统爆炸。
G1_ACTION_SCALE: dict[str, float] = {}
for a in G1_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    G1_ACTION_SCALE[n] = 0.25 * e / s


# 独立运行本脚本时的调试入口，用于在 MuJoCo 原生查看器中可视化 G1 模型
if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_g1_robot_cfg())

  viewer.launch(robot.spec.compile())