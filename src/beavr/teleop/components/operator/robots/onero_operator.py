"""Onero A1 teleoperation operator.

The retargeting pipeline (hand frame -> robot frame, filtering, publishing) is
hardware agnostic and lives in :class:`XArmOperator`. The Onero A1 operator only
supplies the calibration matrices that map the VR / hand-tracking frames onto the
A1 base frame.
"""

from typing import Any, Dict, Optional

import numpy as np

from beavr.teleop.configs.constants import robots

from .xarm7_operator import XArmOperator

# ---------------------------------------------------------------------------------
# Calibration matrices
# ---------------------------------------------------------------------------------
# H_R_V: rotation from the robot base frame to the VR base frame.
# H_T_V: rotation from the hand tracking base frame to the VR base frame.
#
# !!! PLACEHOLDERS !!!
# The matrices below are seeded from the xArm7 desktop setup so that the operator
# can be launched end-to-end. They are NOT calibrated for the Onero A1 cell and
# must be re-derived on the real hardware: engage teleoperation, move the hand
# slowly along the VR left/right/up axes and verify the A1 end-effector follows
# the same directions, then adjust the rotations until the mapping is intuitive.
H_R_V_RIGHT = np.array([[0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
H_T_V_RIGHT = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])

H_R_V_LEFT = np.array([[0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
H_T_V_LEFT = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])


class OneroOperator(XArmOperator):
    """Operator driving an Onero A1 arm from VR hand tracking.

    Inherits the whole retargeting implementation from :class:`XArmOperator` and
    provides the A1 specific calibration matrices plus the operator name.
    """

    _CALIBRATION = {
        robots.RIGHT: (H_R_V_RIGHT, H_T_V_RIGHT),
        robots.LEFT: (H_R_V_LEFT, H_T_V_LEFT),
    }

    def __init__(
        self,
        host: str,
        transformed_keypoints_port: int,
        stream_configs: Dict[str, Any],
        stream_oculus: bool,
        endeff_publish_port: int,
        endeff_subscribe_port: int,
        moving_average_limit: int,
        use_filter: bool = True,
        arm_resolution_port: Optional[int] = None,
        teleoperation_state_port: Optional[int] = None,
        logging_config: Optional[Dict[str, Any]] = None,
        hand_side: str = robots.RIGHT,
    ):
        """
        Args:
            host: Network host address for ZMQ communication.
            transformed_keypoints_port: Port receiving transformed hand keypoints.
            stream_configs: Configuration for streaming data.
            stream_oculus: Flag indicating whether Oculus streaming is used.
            endeff_publish_port: Port publishing end-effector commands.
            endeff_subscribe_port: Port subscribing to the robot's current pose.
            moving_average_limit: Number of samples for the moving average filter.
            use_filter: Whether to enable the complementary state filter.
            arm_resolution_port: Optional port for arm resolution control messages.
            teleoperation_state_port: Optional port for teleoperation state messages.
            logging_config: Optional configuration dictionary for pose logging.
            hand_side: Either ``robots.RIGHT`` or ``robots.LEFT``.
        """
        h_r_v, h_t_v = self._CALIBRATION[hand_side]

        super().__init__(
            operator_name=f"{robots.ROBOT_NAME_ONERO}_{hand_side}_operator",
            host=host,
            transformed_keypoints_port=transformed_keypoints_port,
            stream_configs=stream_configs,
            stream_oculus=stream_oculus,
            endeff_publish_port=endeff_publish_port,
            endeff_subscribe_port=endeff_subscribe_port,
            moving_average_limit=moving_average_limit,
            h_r_v=h_r_v,
            h_t_v=h_t_v,
            use_filter=use_filter,
            arm_resolution_port=arm_resolution_port,
            teleoperation_state_port=teleoperation_state_port,
            logging_config=logging_config,
            hand_side=hand_side,
        )
