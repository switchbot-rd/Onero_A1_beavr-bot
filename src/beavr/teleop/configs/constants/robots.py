# -----------------------------------------------------------------------------
# Robot Configuration Constants
# -----------------------------------------------------------------------------

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


def _float_list_env(name: str, default: list[float]) -> list[float]:
    """Read a comma-separated list of floats from the environment.

    Falls back to ``default`` when unset, and when set to something unparseable
    -- a typo in a launch script should not stop teleop from starting.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return list(default)
    try:
        return [float(part) for part in raw.split(",")]
    except ValueError:
        logger.warning(
            "ignoring %s=%r: expected a comma-separated list of numbers", name, raw
        )
        return list(default)


def _str_env(name: str, default: str) -> str:
    """Read a non-empty string from the environment, else the default.

    A blank value has to mean "unset": for the CAN interface it would otherwise
    reach the adapter as ``""`` and fail with a misleading message, and for the
    network host it would silently point every SUB socket nowhere.
    """
    return os.environ.get(name, "").strip() or default


ROBOT_NAME_LEAP = "leap"
ROBOT_NAME_XARM7 = "xarm7"
ROBOT_NAME_ONERO = "onero"

# -----------------------------------------------------------------------------
# VR detector constants
# -----------------------------------------------------------------------------
# Arm movement
WRIST_HOME_STATE = {
    "translation": [0, 0, 0],
    "rotation_matrix": [1, 0, 0, 0, 1, 0, 0, 0, -1],
}

# -----------------------------------------------------------------------------
# Joint Information
# -----------------------------------------------------------------------------
OCULUS_NUM_KEYPOINTS = 26
VR_THUMB_BOUND_VERTICES = 4
GRIPPER_OPEN = 0
GRIPPER_CLOSE = 1

OCULUS_JOINTS = {
    # Indices follow XRHandJointID order:
    # 0 Wrist, 1 Palm,
    # Thumb: 2 Metacarpal, 3 Proximal, 4 Distal, 5 Tip
    # Index: 6 Metacarpal, 7 Proximal, 8 Intermediate, 9 Distal, 10 Tip
    # Middle: 11 Metacarpal, 12 Proximal, 13 Intermediate, 14 Distal, 15 Tip
    # Ring: 16 Metacarpal, 17 Proximal, 18 Intermediate, 19 Distal, 20 Tip
    # Little: 21 Metacarpal, 22 Proximal, 23 Intermediate, 24 Distal, 25 Tip
    "metacarpals": [2, 6, 11, 16, 21],
    # Knuckles = first finger joints at the hand (proximal joints for non-thumb)
    "knuckles": [7, 12, 17, 22],  # index, middle, ring, little
    # Per-finger chains (4 joints) used with wrist stacked in code -> 5 points total
    "thumb": [2, 3, 4, 5],
    "index": [7, 8, 9, 10],
    "middle": [12, 13, 14, 15],
    "ring": [17, 18, 19, 20],
    "pinky": [22, 23, 24, 25],  # little finger
}

OCULUS_VIEW_LIMITS = {
    "x_limits": [-0.04, 0.04],
    "y_limits": [-0.02, 0.25],
    "z_limits": [-0.04, 0.04],
}

# -----------------------------------------------------------------------------
# XELA Sensor parameters
# -----------------------------------------------------------------------------
XELA_FPS = 100
XELA_NUM_SENSORS = 18  # 3 in thumb 4 in other 3 fingers
XELA_PALM_NUM_SENSORS = 3
XELA_FINGERTIP_NUM_SENSORS = 4
XELA_FINGER_NUM_SENSORS = 11
XELA_PALM_NUM_TAXELS = 24
XELA_FINGERTIP_NUM_TAXELS = 30
XELA_FINGER_NUM_TAXELS = 16
XELA_NUM_TAXELS = 16

# -----------------------------------------------------------------------------
# Robot parameters
# -----------------------------------------------------------------------------
# Allegro and Leap
ALLEGRO_JOINTS_PER_FINGER = 4
LEAP_JOINTS_PER_FINGER = 4
ALLEGRO_JOINT_OFFSETS = {"index": 0, "middle": 4, "ring": 8, "thumb": 12}
LEAP_JOINT_OFFSETS = {"index": 0, "middle": 4, "ring": 8, "thumb": 12}

# -----------------------------------------------------------------------------
# XArm constants
# -----------------------------------------------------------------------------
XARM_SCALE_FACTOR = 1000

# -----------------------------------------------------------------------------
# Onero constants
# -----------------------------------------------------------------------------
ONERO_DEFAULT_MODEL = "A1R"
ONERO_LEFT_DEFAULT_MODEL = "A1L"
ONERO_JOINT_COUNT = 7

# SocketCAN interfaces the A1 arms are wired to. Update these to match the
# deployment machine (any already-UP SocketCAN interface name works).
# SocketCAN interface the A1 is wired to. Overridable from the environment
# because the interface name is a property of the rig, not of this code: a
# native CAN hat enumerates as "can0", while a serial CAN bridge running slcand
# is something like "slcan0". There is no CLI path that reaches these --
# teleop.py builds the robot config from the registry in load_robot_config(),
# which takes only robot_name and laterality.
ONERO_DEFAULT_CAN_CHANNEL = _str_env("ONERO_CAN_CHANNEL", "can0")
ONERO_LEFT_CAN_CHANNEL = _str_env("ONERO_LEFT_CAN_CHANNEL", "can1")

# Installation preset forwarded to the onero SDK.
#
# Do NOT leave these as None. ``resolve_mounting(None)`` silently returns
# UPRIGHT, which places the modelled gravity along base -Z. The A1 is normally
# mounted sideways -- the SDK's own mounting.py says so -- so on a real cell
# that silently rotates the gravity vector by 90 degrees: the feedforward
# pushes sideways instead of up, the joints lag under load, and a servo hold
# against a rigid support becomes a sustained stall (audible screech and motor
# heat rather than useful tracking).
#
# The validated A1 cell in this repo (oneroh1lerobot/configs/*.json) uses
# humanoid_left / humanoid_right with with_gripper=true. Those presets put
# base +X up and base +Z horizontally outward, so the first joint swings the
# whole arm in the vertical plane. Keep the two arms mirrored.
ONERO_DEFAULT_MOUNTING = "humanoid_right"
ONERO_LEFT_DEFAULT_MOUNTING = "humanoid_left"

# The A1 teleoperation arms ship with a gripper fitted, and the SDK's gripper
# gravity profile is required in that case: it scales the modelled per-joint
# gravity torque by up to 5.3x (J5) and 2.8x (J3). Set to False only for a bare
# flange — over-compensation is worse than under-compensation, so do not leave
# this True on a gripper-less arm.
ONERO_DEFAULT_WITH_GRIPPER = True

# A safe "park" pose for the right arm on the cell this repo was developed
# against, in degrees, in SDK joint order.
#
# Derived, not guessed. Arm.home() drives every joint to zero, and on that cell
# the all-zero pose puts the lowest link 247 mm BELOW the plane the arm rests on;
# the straight-line path there crosses that plane after only ~5% of the motion.
# This pose clears the table by 60 mm instead, and reaching it from a resting
# pose costs just 6.2 deg of joint travel -- the whole path stays above the
# table. Found by maximising clearance-minus-travel over the URDF and its
# collision meshes.
#
# Caveat: that search modelled the table plane only. Confirm visually that it
# clears the mounting post and any fixtures before relying on it.
# Overridable from the environment (ONERO_PARK_POSE_DEG="deg,deg,...") because the
# pose belongs to the cell rather than to this code, and no CLI route reaches it:
# load_robot_config() takes only robot_name and laterality.
ONERO_SAFE_PARK_POSE_DEG = _float_list_env(
    "ONERO_PARK_POSE_DEG", [17.14, 6.22, -5.48, 70.22, -2.19, -18.58, 0.80]
)

# XArm home poses
BIMANUAL_LEFT_HOME = [206, 0, 475, 3.142, 0, 0]
BIMANUAL_RIGHT_HOME = [206, 0, 475, 3.142, 0, 0]
ROBOT_HOME_POSE_AA = [206.0, -0.0, 475, 3.142, 0.0, 0.0]
ROBOT_HOME_JS = [
    0.0,
    -0.4363323129985824,
    -0.017453292519943295,
    0.4537856055185257,
    0.0,
    0.8726646259971648,
    0.0,
]

# -----------------------------------------------------------------------------
# LEAP hand solver scaling factors (used by LeapHandIKSolver)
# -----------------------------------------------------------------------------
LEAP_FINGER_SCALE_FACTOR = 1.8  # default scaling for non-thumb fingertips
LEAP_THUMB_SCALE_FACTOR = 1.7  # separate scaling for thumb positions
LEAP_HOME_JS = np.zeros(16)

# -----------------------------------------------------------------------------
# Oculus constants
# -----------------------------------------------------------------------------
VR_DETECTOR = "vr detector"
KEYPOINTS = "keypoints"
BUTTON = "button"
PAUSE = "pause"
RESUME = "resume"
RIGHT = "right"
LEFT = "left"
BIMANUAL = "bimanual"

# -----------------------------------------------------------------------------
# Keypoint transform constants
# -----------------------------------------------------------------------------
KEYPOINT_POSITION_TRANSFORM = "keypoint position transform"
ABSOLUTE = "absolute"
RELATIVE = "relative"
TRANSFORMED_HAND_COORDS = "transformed_hand_coords"
TRANSFORMED_HAND_FRAME = "transformed_hand_frame"

# -----------------------------------------------------------------------------
# Data recording parameters
# -----------------------------------------------------------------------------
ALLEGRO_SAMPLE_OFFSET = 10  # For sampling states
SAMPLE_WRITER_FPS = 5

# -----------------------------------------------------------------------------
# Robot configuration flags
# -----------------------------------------------------------------------------
# To run the robot
OPERATE = True
# To run the robot interface
ROBOT_INTERFACE = True
# To run the simulation environment
SIM_ENV = False
# To run the Xela controller
RUN_XELA = False

# Visualize 3D and 2D plots
VISUALIZE_XELA = False
VISUALIZE_RIGHT_2D = False
VISUALIZE_RIGHT_3D = False
VISUALIZE_RIGHT_DIR = False

# -----------------------------------------------------------------------------
# Robot identifiers for recording
# -----------------------------------------------------------------------------
ROBOT_IDENTIFIER_RIGHT_XARM7 = "right_xarm7"
ROBOT_IDENTIFIER_LEFT_XARM7 = "left_xarm7"
ROBOT_IDENTIFIER_RIGHT_ONERO = "right_onero"
ROBOT_IDENTIFIER_LEFT_ONERO = "left_onero"
ROBOT_IDENTIFIER_RIGHT_LEAP_HAND = "right_leap"
ROBOT_IDENTIFIER_LEFT_LEAP_HAND = "left_leap"
ROBOT_IDENTIFIER_LEAP = "leap"

# -----------------------------------------------------------------------------
# Recorded data types
# -----------------------------------------------------------------------------
RECORDED_DATA_JOINT_STATES = "joint_states"
RECORDED_DATA_CARTESIAN_STATES = "cartesian_states"
RECORDED_DATA_XARM_CARTESIAN_STATES = "xarm_cartesian_states"
RECORDED_DATA_ONERO_CARTESIAN_STATES = "onero_cartesian_states"
RECORDED_DATA_COMMANDED_CARTESIAN_STATE = "commanded_cartesian_state"
RECORDED_DATA_COMMANDED_JOINT_STATES = "commanded_joint_states"
RECORDED_DATA_JOINT_ANGLES_RAD = "joint_angles_rad"

# -----------------------------------------------------------------------------
# VR frequency constants
# -----------------------------------------------------------------------------
VR_FREQ = 30
RECORDER_FREQ = 30
ARM_TELEOP_CONT = 1
ARM_TELEOP_STOP = 0
ARM_HIGH_RESOLUTION = 1
ARM_LOW_RESOLUTION = 0
TELEOP_HANDSHAKE_PORT = 8150

# -----------------------------------------------------------------------------
# VR display constants
# -----------------------------------------------------------------------------
# Calibration file paths
CALIBRATION_FILES_PATH = "calibration_files"
VR_THUMB_BOUNDS_PATH = os.path.join(CALIBRATION_FILES_PATH, "vr_thumb_bounds.npy")
VR_DISPLAY_THUMB_BOUNDS_PATH = os.path.join(CALIBRATION_FILES_PATH, "vr_thumb_plot_bounds.npy")
VR_2D_PLOT_SAVE_PATH = os.path.join(CALIBRATION_FILES_PATH, "oculus_hand_2d_plot.jpg")
XELA_PLOT_SAVE_PATH = os.path.join(CALIBRATION_FILES_PATH, "xela_plot.png")
