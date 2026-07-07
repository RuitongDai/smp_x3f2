"""F2 14DOF robot constants.

这个文件用于把 F2 14 自由度 XML 注册成 mjlab 可用的 EntityCfg。
当前版本使用固定上身模型，只控制双腿 12 个关节 + 腰部 2 个关节。
"""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

# =============================================================================
# MJCF 路径和 mesh 资源
# =============================================================================
SRC_PATH = Path(__file__).parent.parent
# F2 14DOF XML 路径。
X3F2_XML: Path = (
  SRC_PATH / "x3" / "xmls" / "x3f2_14dof.xml"
)

X3F2_MESH_DIR: Path = X3F2_XML.parent / "meshes"

assert X3F2_XML.exists(), f"F2 XML file not found: {X3F2_XML}"
assert X3F2_MESH_DIR.exists(), f"F2 mesh dir not found: {X3F2_MESH_DIR}"

def get_spec() -> mujoco.MjSpec:
  """从 XML 文件加载并返回 MuJoCo 的 MjSpec 对象。"""
  return mujoco.MjSpec.from_file(str(X3F2_XML))

# =============================================================================
# Actuator 配置
# =============================================================================


X3F2_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint",),
  stiffness=100.0,
  damping=2.0,
  effort_limit=75.0,
  armature=0.01,
)


X3F2_ACTUATOR_HIP_ROLL_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_roll_joint",
    ".*_hip_yaw_joint",
  ),
  stiffness=100.0,
  damping=2.0,
  effort_limit=87.0,
  armature=0.01,
)

X3F2_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_knee_joint",),
  stiffness=100.0,
  damping=2.0,
  effort_limit=120.0,
  armature=0.01,
)

X3F2_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint",),
  stiffness=30.0,
  damping=2.0,
  effort_limit=89.0,
  armature=0.01,
)

X3F2_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_roll_joint",),
  stiffness=30.0,
  damping=2.0,
  effort_limit=12.0,
  armature=0.01,
)

X3F2_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "waist_yaw_joint",
    "waist_roll_joint",
  ),
  stiffness=100.0,
  damping=2.0,
  effort_limit=87.0,
  armature=0.01,
)


X3F2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    X3F2_ACTUATOR_HIP_PITCH,
    X3F2_ACTUATOR_HIP_ROLL_YAW,
    X3F2_ACTUATOR_KNEE,
    X3F2_ACTUATOR_ANKLE_PITCH,
    X3F2_ACTUATOR_ANKLE_ROLL,
    X3F2_ACTUATOR_WAIST,
  ),
  soft_joint_pos_limit_factor=0.9,
)

# =============================================================================
# 初始姿态
# =============================================================================

HOME_KEY_FRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.873),
  joint_pos={
    'left_hip_pitch_joint': -0.10,
    'left_hip_roll_joint': 0,
    'left_hip_yaw_joint': 0.,
    'left_knee_joint': 0.20,
    'left_ankle_pitch_joint': -0.10,
    'left_ankle_roll_joint': 0,
    'right_hip_pitch_joint': -0.10,
    'right_hip_roll_joint': 0,
    'right_hip_yaw_joint': 0.,
    'right_knee_joint': 0.20,
    'right_ankle_pitch_joint': -0.10,
    'right_ankle_roll_joint': 0,
    'waist_roll_joint': 0,
    'waist_yaw_joint': 0,
  },
  joint_vel={".*": 0.0},
)

# =============================================================================
# Collision 配置
# =============================================================================

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-4]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-4]_collision$": 1}, # 脚部碰撞优先级更高
  friction={r"^(left|right)_foot[1-4]_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-4]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-4]_collision$": 1},
  friction={r"^(left|right)_foot[1-4]_collision$": (0.6,)},
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-4]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)


# =============================================================================
# EntityCfg 接口
# =============================================================================

def get_x3f2_robot_cfg() -> EntityCfg:
  """返回 F2 14DOF 的 mjlab EntityCfg。"""
  return EntityCfg(
    init_state=HOME_KEY_FRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=X3F2_ARTICULATION,
  )


# =============================================================================
# Action scale
# =============================================================================

X3F2_ACTION_SCALE: dict[str, float] = {}

for actuator in X3F2_ARTICULATION.actuators:
  assert isinstance(actuator, BuiltinPositionActuatorCfg)

  effort_limit = actuator.effort_limit
  stiffness = actuator.stiffness
  target_names_expr = actuator.target_names_expr

  assert effort_limit is not None
  assert stiffness is not None

  for name_expr in target_names_expr:
    X3F2_ACTION_SCALE[name_expr] = 0.25 * effort_limit / stiffness


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_x3f2_robot_cfg())
  viewer.launch(robot.spec.compile())