"""Tests for the Onero A1 teleoperation operator."""

import time

import numpy as np

from beavr.teleop.components.detector.detector_types import InputFrame
from beavr.teleop.components.interface.interface_types import CartesianState
from beavr.teleop.components.operator.operator_types import CartesianTarget
from beavr.teleop.components.operator.robots.onero_operator import OneroOperator
from beavr.teleop.configs.constants import robots


def _make_operator(hand_side=robots.RIGHT, transformed_keypoints_port=5555, endeff_publish_port=10032):
    return OneroOperator(
        host="127.0.0.1",
        transformed_keypoints_port=transformed_keypoints_port,
        stream_configs={},
        stream_oculus=False,
        endeff_publish_port=endeff_publish_port,
        endeff_subscribe_port=10031,
        moving_average_limit=3,
        use_filter=False,
        arm_resolution_port=None,
        teleoperation_state_port=None,
        logging_config={"enabled": False},
        hand_side=hand_side,
    )


def test_operator_name_and_calibration_per_side():
    right = _make_operator(robots.RIGHT)
    left = _make_operator(robots.LEFT)

    assert right.operator_name == "onero_right_operator"
    assert left.operator_name == "onero_left_operator"
    np.testing.assert_allclose(right.h_r_v, OneroOperator._CALIBRATION[robots.RIGHT][0])
    np.testing.assert_allclose(left.h_t_v, OneroOperator._CALIBRATION[robots.LEFT][1])


def test_operator_reuses_xarm_retargeting_pipeline(bus):
    """A static hand frame must produce a CartesianTarget on 'endeff_coords'."""
    transformed_keypoints_port = 5555
    endeff_publish_port = 10032
    endeff_subscribe_port = 10031

    identity = np.eye(4, dtype=np.float64)
    op = OneroOperator(
        host="127.0.0.1",
        transformed_keypoints_port=transformed_keypoints_port,
        stream_configs={},
        stream_oculus=False,
        endeff_publish_port=endeff_publish_port,
        endeff_subscribe_port=endeff_subscribe_port,
        moving_average_limit=3,
        use_filter=False,
        arm_resolution_port=None,
        teleoperation_state_port=None,
        logging_config={"enabled": False},
        hand_side=robots.RIGHT,
    )

    # Robot pose reply used as the teleop reset baseline.
    bus.publish(
        "127.0.0.1",
        endeff_subscribe_port,
        "endeff_homo",
        CartesianState(timestamp_s=time.time(), h_matrix=tuple(map(tuple, np.eye(4).tolist()))),
    )

    frame_vectors = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    right_frame_topic = f"{robots.RIGHT}_{robots.TRANSFORMED_HAND_FRAME}"
    input_frame = InputFrame(
        timestamp_s=time.time(),
        hand_side=robots.RIGHT,
        keypoints=[(0.0, 0.0, 0.0)] * robots.OCULUS_NUM_KEYPOINTS,
        is_relative=False,
        frame_vectors=frame_vectors,
    )

    bus.publish("127.0.0.1", transformed_keypoints_port, right_frame_topic, input_frame)
    op.h_r_v = identity
    op.h_t_v = identity
    op._apply_retargeted_angles()

    bus.publish("127.0.0.1", transformed_keypoints_port, right_frame_topic, input_frame)
    op._apply_retargeted_angles()

    cmd = bus.recv_latest(endeff_publish_port, "endeff_coords")
    assert isinstance(cmd, CartesianTarget)
    np.testing.assert_allclose(np.asarray(cmd.position_m), np.zeros(3), atol=1e-6)
    quat = np.asarray(cmd.orientation_xyzw)
    assert np.isclose(np.linalg.norm(quat), 1.0, atol=1e-6)
