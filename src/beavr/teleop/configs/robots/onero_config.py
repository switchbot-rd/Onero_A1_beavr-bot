"""Config for the Onero A1 arm, supporting right, left or bimanual laterality.

The A1 is wired over SocketCAN and driven through the ``onero-sdk`` package, so
each arm is addressed by its CAN interface name instead of an IP address.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from beavr.teleop.common.configs.loader import Laterality, log_laterality_configuration
from beavr.teleop.components.detector.vr.onero_oculus import OneroVRHandDetector
from beavr.teleop.components.interface.robots.onero_robot import OneroRobot

# Import shared constants and utilities
from beavr.teleop.configs.constants import network, ports, robots
from beavr.teleop.configs.robots import TeleopRobotConfig

# Import shared component configurations
from beavr.teleop.configs.robots.shared_components import (
    SharedComponentRegistry,
    UnifiedOculusVRHandDetectorCfg,
)

logger = logging.getLogger(__name__)


@dataclass
class OneroRobotCfg:
    host: str = network.HOST_ADDRESS
    can_channel: str = robots.ONERO_DEFAULT_CAN_CHANNEL
    model: str = robots.ONERO_DEFAULT_MODEL
    mounting: Optional[str] = None
    world_frame: bool = True
    # The A1 teleoperation arms ship with a gripper; its gravity profile scales
    # the modelled gravity torque by up to 5.3x (J5) / 2.8x (J3). Keep False only
    # for a bare flange, otherwise the joints sag under load.
    with_gripper: bool = False
    # Onero's own gripper (commands on CAN id 0x08, feedback 0x18) shares the arm
    # bus. Enabling it attaches a *controllable* effector -- ``with_gripper`` only
    # scales the gravity model and cannot move anything.
    gripper_enabled: bool = True
    gripper_command_port: int = 10040  # +2 for the left arm
    is_right_arm: bool = True  # Overridden based on laterality
    endeff_publish_port: int = ports.ONERO_ENDEFF_PUBLISH_PORT
    endeff_subscribe_port: int = ports.ONERO_ENDEFF_SUBSCRIBE_PORT
    joint_subscribe_port: int = ports.ONERO_JOINT_SUBSCRIBE_PORT
    reset_subscribe_port: int = ports.ONERO_RESET_SUBSCRIBE_PORT
    home_subscribe_port: int = ports.ONERO_HOME_SUBSCRIBE_PORT
    state_publish_port: int = ports.ONERO_STATE_PUBLISH_PORT
    teleoperation_state_port: int = ports.ONERO_TELEOPERATION_STATE_PORT
    hand_side: str = robots.RIGHT  # Overridden based on laterality
    # Position/damping gains forwarded to the SDK servo. None inherits the SDK's
    # own defaults (uniform kp=50, kd=1.0), which is also what the validated
    # deployment uses. Leave it that way.
    #
    # No scalar pair is actually right here, because servoj broadcasts one value
    # to all seven joints while the model's calibrated gains are per-joint:
    #
    #   default_kp_values = [150, 150, 150, 150,  40,  40,  40]
    #   default_kd_values = [  4,   4,   4,   4,   1,   1,   1]
    #
    # Measured on this bench at 100 Hz:
    #
    #   kp=50,  kd=1 -> proximal joints sit below their friction breakaway band
    #                   (J1 carries ~1.5 Nm, so 1.5/50 = 1.72 deg is needed to
    #                   move it), so only 2 of 7 joints tracked and the Cartesian
    #                   error held at 11.8 mm.
    #   kp=150, kd=1 -> that band drops 3x, 5 of 7 joints tracked and the error
    #                   fell to 4.8 mm -- but the proximal joints are then
    #                   under-damped, since their calibrated kd is 4.0 and not
    #                   1.0, and the arm oscillated audibly in teleop.
    #   kp=150, kd=4 -> the wrist instead gets 4x its calibrated damping of 1.0
    #                   and screeches.
    #
    # So the SDK's scalar 50/1.0 is the safest compromise and the real fix is
    # per-joint gains. That needs an API-layer change only: the backend and wire
    # protocol are already per-joint (SafeBackend.command takes kps/kds lists,
    # MIT frames encode them per joint, and the movej family already runs a
    # per-joint GainPolicy). The registry values above exist but are currently
    # wired only into speedj/speedl through _velocity_gains().
    #
    # Left at None deliberately. Do not "fix" this by wiring a scalar from the
    # environment: measured on this bench, every flat pair is wrong somewhere.
    kp: Optional[float] = None
    kd: Optional[float] = None
    # Pose the control loop sends the arm to on a home command, in degrees, in
    # SDK joint order. Defaults to the pose derived for the development cell
    # (see ONERO_SAFE_PARK_POSE_DEG); override per cell. Without it, home() falls
    # back to Arm.home(), which drives every joint to zero -- 247 mm below the
    # table on that cell.
    park_pose_deg: Optional[list[float]] = field(
        default_factory=lambda: list(robots.ONERO_SAFE_PARK_POSE_DEG)
    )
    # Travel the arm to park_pose_deg before the control loop tracks. OFF by
    # default, so starting teleop never moves the hardware on its own. See
    # OneroRobot._startup_home.
    home_on_start: bool = False
    recorder_config: dict[str, Any] = field(
        default_factory=lambda: {
            "robot_identifier": robots.ROBOT_IDENTIFIER_RIGHT_ONERO,
            "recorded_data": [
                robots.RECORDED_DATA_JOINT_STATES,
                robots.RECORDED_DATA_ONERO_CARTESIAN_STATES,
                robots.RECORDED_DATA_COMMANDED_CARTESIAN_STATE,
                robots.RECORDED_DATA_JOINT_ANGLES_RAD,
            ],
        }
    )

    def __post_init__(self):
        """Validate configuration."""
        all_ports = [
            self.endeff_publish_port,
            self.endeff_subscribe_port,
            self.joint_subscribe_port,
            self.reset_subscribe_port,
            self.state_publish_port,
            self.home_subscribe_port,
            self.teleoperation_state_port,
        ]
        for port in all_ports:
            if not (1 <= port <= 65535):
                raise ValueError(f"Port out of valid range (1-65535): {port}")

        if not self.can_channel:
            raise ValueError("Onero A1 requires a SocketCAN interface name (e.g. 'can0')")

        # A wrong-length park pose only surfaces as a refused home command far
        # later, so catch it where the mistake is made.
        if self.park_pose_deg is not None and len(self.park_pose_deg) != robots.ONERO_JOINT_COUNT:
            raise ValueError(
                f"park_pose_deg needs {robots.ONERO_JOINT_COUNT} joint angles in "
                f"degrees, got {len(self.park_pose_deg)}"
            )

    def build(self):
        return OneroRobot(
            host=self.host,
            can_channel=self.can_channel,
            model=self.model,
            mounting=self.mounting,
            world_frame=self.world_frame,
            with_gripper=self.with_gripper,
            gripper_enabled=self.gripper_enabled,
            gripper_command_port=self.gripper_command_port,
            is_right_arm=self.is_right_arm,
            endeff_publish_port=self.endeff_publish_port,
            endeff_subscribe_port=self.endeff_subscribe_port,
            joint_subscribe_port=self.joint_subscribe_port,
            reset_subscribe_port=self.reset_subscribe_port,
            home_subscribe_port=self.home_subscribe_port,
            state_publish_port=self.state_publish_port,
            teleoperation_state_port=self.teleoperation_state_port,
            kp=self.kp,
            kd=self.kd,
            home_on_start=self.home_on_start,
            park_pose_deg=self.park_pose_deg,
            recorder_config=self.recorder_config,
        )


@dataclass
class OneroOperatorCfg:
    host: str = network.HOST_ADDRESS
    transformed_keypoints_port: int = ports.KEYPOINT_TRANSFORM_PORT
    stream_configs: dict[str, Any] = field(
        default_factory=lambda: {
            "host": network.HOST_ADDRESS,
            "port": ports.CONTROL_STREAM_PORT,
        }
    )
    stream_oculus: bool = True
    endeff_publish_port: int = ports.ONERO_ENDEFF_SUBSCRIBE_PORT
    endeff_subscribe_port: int = ports.ONERO_ENDEFF_PUBLISH_PORT
    moving_average_limit: int = 3
    arm_resolution_port: int = ports.KEYPOINT_STREAM_PORT
    use_filter: bool = False
    # Carries the App's start/stop signal, which the detector republishes on its
    # keypoint port. 10038 (ONERO_TELEOPERATION_STATE_PORT) has no publisher on this
    # rig, which is why the operator used to sit in CONT from the first frame.
    teleoperation_state_port: int = ports.KEYPOINT_STREAM_PORT
    logging_config: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "log_dir": "logs",
            "log_poses": True,
            "log_prefix": "onero",
        }
    )
    hand_side: str = robots.RIGHT  # Overridden based on laterality
    # Port the operator publishes two-state gripper commands on. None disables the
    # operator side of the chain (the robot then never receives a command).
    gripper_command_port: Optional[int] = None

    def __post_init__(self):
        """Validate configuration."""
        all_ports = [
            self.transformed_keypoints_port,
            self.endeff_publish_port,
            self.endeff_subscribe_port,
            self.arm_resolution_port,
            self.teleoperation_state_port,
        ]
        for port in all_ports:
            if not (1 <= port <= 65535):
                raise ValueError(f"Port out of valid range (1-65535): {port}")

        if self.moving_average_limit < 1:
            raise ValueError(f"moving_average_limit must be >= 1: {self.moving_average_limit}")

    def build(self):
        # Import here to avoid circular imports
        from beavr.teleop.components.operator.robots.onero_operator import OneroOperator

        return OneroOperator(
            host=self.host,
            transformed_keypoints_port=self.transformed_keypoints_port,
            stream_configs=self.stream_configs,
            stream_oculus=self.stream_oculus,
            endeff_publish_port=self.endeff_publish_port,
            endeff_subscribe_port=self.endeff_subscribe_port,
            moving_average_limit=self.moving_average_limit,
            arm_resolution_port=self.arm_resolution_port,
            use_filter=self.use_filter,
            teleoperation_state_port=self.teleoperation_state_port,
            logging_config=self.logging_config,
            hand_side=self.hand_side,
            gripper_command_port=self.gripper_command_port,
        )


@dataclass
class OneroDetectorCfg(UnifiedOculusVRHandDetectorCfg):
    """Detector config that builds the Onero frame-validating detector.

    Everything else is inherited, so the ports and hand routing stay identical to
    the shared configuration; only the parser is specialised.
    """

    def build(self):
        kwargs = {
            "host": self.host,
            "oculus_pub_port": self.oculus_pub_port,
            "button_port": self.button_port,
            "teleop_reset_port": self.teleop_reset_port,
            "hand_config": self.hand_config,
        }
        if self.hand_config in [robots.RIGHT, robots.BIMANUAL]:
            kwargs["right_hand_port"] = ports.RIGHT_HAND_OCULUS_RECEIVER_PORT
        if self.hand_config in [robots.LEFT, robots.BIMANUAL]:
            kwargs["left_hand_port"] = ports.LEFT_HAND_OCULUS_RECEIVER_PORT
        return OneroVRHandDetector(**kwargs)


@dataclass
@TeleopRobotConfig.register_subclass(robots.ROBOT_NAME_ONERO)
class OneroA1Config:
    robot_name: str = robots.ROBOT_NAME_ONERO
    laterality: Laterality = Laterality.RIGHT

    # Configuration components - populated based on laterality
    detector: list = field(default_factory=list)
    transforms: list = field(default_factory=list)
    visualizers: list = field(default_factory=list)
    robots: list = field(default_factory=list)
    operators: list = field(default_factory=list)

    def __post_init__(self):
        """Configure components based on laterality setting."""
        log_laterality_configuration(self.laterality, robots.ROBOT_NAME_ONERO)
        self._configure_for_laterality()

    def _configure_for_laterality(self):
        """Configure all components based on the laterality setting."""
        # Detector -------------------------------------------------------------------
        self.detector = []
        if self.laterality == Laterality.BIMANUAL:
            self.detector.append(
                OneroDetectorCfg(
                    host=network.HOST_ADDRESS,
                    hand_config=robots.BIMANUAL,
                    hand_side="bimanual",
                )
            )
        else:
            hand_side = robots.RIGHT if self.laterality == Laterality.RIGHT else robots.LEFT
            self.detector.append(
                OneroDetectorCfg(
                    host=network.HOST_ADDRESS,
                    hand_config=hand_side,
                    hand_side=hand_side,
                )
            )

        # Keypoint transforms --------------------------------------------------------
        self.transforms = []
        if self.laterality in [Laterality.RIGHT, Laterality.BIMANUAL]:
            self.transforms.append(
                SharedComponentRegistry.get_transform_config(
                    hand_side=robots.RIGHT,
                    host=network.HOST_ADDRESS,
                    keypoint_sub_port=ports.KEYPOINT_STREAM_PORT,
                    moving_average_limit=3,
                )
            )
        if self.laterality in [Laterality.LEFT, Laterality.BIMANUAL]:
            self.transforms.append(
                SharedComponentRegistry.get_transform_config(
                    hand_side=robots.LEFT,
                    host=network.HOST_ADDRESS,
                    keypoint_sub_port=ports.KEYPOINT_STREAM_PORT,
                    moving_average_limit=3,
                )
            )

        # Visualizers ----------------------------------------------------------------
        self.visualizers = []

        # Robots ---------------------------------------------------------------------
        # The interface comes from the constants, which are overridable through
        # ONERO_CAN_CHANNEL / ONERO_LEFT_CAN_CHANNEL. There is no other route:
        # OneroA1Config has no channel field of its own, and load_robot_config()
        # takes only robot_name and laterality. A rig on a serial CAN bridge
        # (slcand, e.g. "slcan0") depends entirely on that override -- without it
        # the adapter tries can0 and _create_arm dies with a connection error.
        right_channel = robots.ONERO_DEFAULT_CAN_CHANNEL
        left_channel = robots.ONERO_LEFT_CAN_CHANNEL
        if self.laterality is Laterality.BIMANUAL and right_channel == left_channel:
            raise ValueError(
                f"both A1 arms resolve to CAN channel {right_channel!r}; set "
                f"ONERO_CAN_CHANNEL and ONERO_LEFT_CAN_CHANNEL to separate buses"
            )

        self.robots = []
        if self.laterality in [Laterality.RIGHT, Laterality.BIMANUAL]:
            self.robots.append(
                OneroRobotCfg(
                    host=network.HOST_ADDRESS,
                    can_channel=right_channel,
                    model=robots.ONERO_DEFAULT_MODEL,
                    mounting=robots.ONERO_DEFAULT_MOUNTING,
                    with_gripper=robots.ONERO_DEFAULT_WITH_GRIPPER,
                    gripper_enabled=robots.ONERO_DEFAULT_WITH_GRIPPER,
                    gripper_command_port=10040,
                    is_right_arm=True,
                    endeff_publish_port=ports.ONERO_ENDEFF_PUBLISH_PORT,
                    endeff_subscribe_port=ports.ONERO_ENDEFF_SUBSCRIBE_PORT,
                    joint_subscribe_port=ports.ONERO_JOINT_SUBSCRIBE_PORT,
                    reset_subscribe_port=ports.ONERO_RESET_SUBSCRIBE_PORT,
                    home_subscribe_port=ports.ONERO_HOME_SUBSCRIBE_PORT,
                    state_publish_port=ports.ONERO_STATE_PUBLISH_PORT,
                    # See the operator note: pause is intentionally left inert.
                    teleoperation_state_port=ports.ONERO_TELEOPERATION_STATE_PORT,
                    hand_side=robots.RIGHT,
                    recorder_config={
                        "robot_identifier": robots.ROBOT_IDENTIFIER_RIGHT_ONERO,
                        "recorded_data": [
                            robots.RECORDED_DATA_JOINT_STATES,
                            robots.RECORDED_DATA_ONERO_CARTESIAN_STATES,
                            robots.RECORDED_DATA_COMMANDED_CARTESIAN_STATE,
                            robots.RECORDED_DATA_JOINT_ANGLES_RAD,
                        ],
                    },
                )
            )
        if self.laterality in [Laterality.LEFT, Laterality.BIMANUAL]:
            self.robots.append(
                OneroRobotCfg(
                    host=network.HOST_ADDRESS,
                    can_channel=left_channel,
                    model=robots.ONERO_LEFT_DEFAULT_MODEL,
                    mounting=robots.ONERO_LEFT_DEFAULT_MOUNTING,
                    with_gripper=robots.ONERO_DEFAULT_WITH_GRIPPER,
                    gripper_enabled=robots.ONERO_DEFAULT_WITH_GRIPPER,
                    gripper_command_port=10042,
                    is_right_arm=False,
                    endeff_publish_port=ports.ONERO_ENDEFF_PUBLISH_PORT + 2,
                    endeff_subscribe_port=ports.ONERO_ENDEFF_SUBSCRIBE_PORT + 2,
                    joint_subscribe_port=ports.ONERO_JOINT_SUBSCRIBE_PORT + 2,
                    reset_subscribe_port=ports.ONERO_RESET_SUBSCRIBE_PORT + 2,
                    home_subscribe_port=ports.ONERO_HOME_SUBSCRIBE_PORT,
                    state_publish_port=ports.ONERO_STATE_PUBLISH_PORT + 2,
                    teleoperation_state_port=ports.ONERO_TELEOPERATION_STATE_PORT,  # see right arm note
                    hand_side=robots.LEFT,
                    recorder_config={
                        "robot_identifier": robots.ROBOT_IDENTIFIER_LEFT_ONERO,
                        "recorded_data": [
                            robots.RECORDED_DATA_JOINT_STATES,
                            robots.RECORDED_DATA_ONERO_CARTESIAN_STATES,
                            robots.RECORDED_DATA_COMMANDED_CARTESIAN_STATE,
                            robots.RECORDED_DATA_JOINT_ANGLES_RAD,
                        ],
                    },
                )
            )

        # Operators ------------------------------------------------------------------
        self.operators = []
        if self.laterality in [Laterality.RIGHT, Laterality.BIMANUAL]:
            self.operators.append(
                OneroOperatorCfg(
                    host=network.HOST_ADDRESS,
                    transformed_keypoints_port=ports.KEYPOINT_TRANSFORM_PORT,
                    endeff_publish_port=ports.ONERO_ENDEFF_SUBSCRIBE_PORT,
                    endeff_subscribe_port=ports.ONERO_ENDEFF_PUBLISH_PORT,
                    moving_average_limit=3,
                    arm_resolution_port=ports.KEYPOINT_STREAM_PORT,
                    use_filter=False,
                    # Point the pause subscriber at the detector's keypoint port:
                    # that is where the App's start/stop signal is republished on
                    # topic "pause". OneroOperator compensates for the detector's
                    # inverted codes and boots in STOP, so teleop now waits for the
                    # App's Start button instead of tracking as soon as it launches.
                    teleoperation_state_port=ports.KEYPOINT_STREAM_PORT,
                    hand_side=robots.RIGHT,
                    gripper_command_port=10040,
                    logging_config={
                        "enabled": False,
                        "log_dir": "logs",
                        "log_poses": True,
                        "log_prefix": "onero_right",
                    },
                )
            )
        if self.laterality in [Laterality.LEFT, Laterality.BIMANUAL]:
            self.operators.append(
                OneroOperatorCfg(
                    host=network.HOST_ADDRESS,
                    transformed_keypoints_port=ports.LEFT_KEYPOINT_TRANSFORM_PORT,
                    endeff_publish_port=ports.ONERO_ENDEFF_SUBSCRIBE_PORT + 2,
                    endeff_subscribe_port=ports.ONERO_ENDEFF_PUBLISH_PORT + 2,
                    moving_average_limit=3,
                    arm_resolution_port=ports.KEYPOINT_STREAM_PORT,
                    use_filter=False,
                    teleoperation_state_port=ports.KEYPOINT_STREAM_PORT,  # see right arm note
                    hand_side=robots.LEFT,
                    gripper_command_port=10042,
                    logging_config={
                        "enabled": False,
                        "log_dir": "logs",
                        "log_poses": True,
                        "log_prefix": "onero_left",
                    },
                )
            )

    def build(self):
        """Build the robot configuration components."""
        return {
            "robot_name": self.robot_name,
            "detector": [detector.build() for detector in self.detector],
            "transforms": [item.build() for item in self.transforms],
            "visualizers": [item.build() for item in self.visualizers],
            "robots": [item.build() for item in self.robots],
            "operators": [item.build() for item in self.operators],
        }
