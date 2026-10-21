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
    # The reset inside this call derives the calibration pair from the baselines
    # above (identity flange pose, identity hand frame), so nothing is preset here.
    op._apply_retargeted_angles()

    bus.publish("127.0.0.1", transformed_keypoints_port, right_frame_topic, input_frame)
    op._apply_retargeted_angles()

    cmd = bus.recv_latest(endeff_publish_port, "endeff_coords")
    assert isinstance(cmd, CartesianTarget)
    np.testing.assert_allclose(np.asarray(cmd.position_m), np.zeros(3), atol=1e-6)
    quat = np.asarray(cmd.orientation_xyzw)
    assert np.isclose(np.linalg.norm(quat), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------------
# Calibration derived at reset
# ---------------------------------------------------------------------------------


def _publish_robot_pose(bus, port, rotation, position=(0.0, 0.0, 0.0)):
    h = np.eye(4, dtype=np.float64)
    h[:3, :3] = np.asarray(rotation, dtype=float)
    h[:3, 3] = np.asarray(position, dtype=float)
    bus.publish(
        "127.0.0.1",
        port,
        "endeff_homo",
        CartesianState(timestamp_s=time.time(), h_matrix=tuple(map(tuple, h.tolist()))),
    )


def _publish_hand_frame(bus, port, basis, origin=(0.0, 0.0, 0.0)):
    vectors = (tuple(float(v) for v in origin),) + tuple(
        tuple(float(v) for v in np.asarray(basis, dtype=float)[:, i]) for i in range(3)
    )
    bus.publish(
        "127.0.0.1",
        port,
        f"{robots.RIGHT}_{robots.TRANSFORMED_HAND_FRAME}",
        InputFrame(
            timestamp_s=time.time(),
            hand_side=robots.RIGHT,
            keypoints=[(0.0, 0.0, 0.0)] * robots.OCULUS_NUM_KEYPOINTS,
            is_relative=False,
            frame_vectors=vectors,
        ),
    )


def _rot_x(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _unity_to_robot():
    from beavr.teleop.components.operator.robots.onero_operator import _UNITY_TO_ROBOT

    return _UNITY_TO_ROBOT


def test_reset_derives_calibration_instead_of_keeping_the_nominal_pair(bus):
    op = _make_operator(transformed_keypoints_port=5555, endeff_publish_port=10032)
    nominal = op.h_t_v.copy()

    flange = _rot_x(np.deg2rad(35.0))
    hand = np.diag([-1.0, 1.0, -1.0])
    _publish_robot_pose(bus, 10031, flange)
    _publish_hand_frame(bus, 5555, hand)
    op._apply_retargeted_angles()

    expected_r, expected_t = OneroOperator._calibration_from_baselines(
        np.block([[flange, np.zeros((3, 1))], [0, 0, 0, 1]]),
        np.block([[hand, np.zeros((3, 1))], [0, 0, 0, 1]]),
    )
    np.testing.assert_allclose(op.h_r_v, expected_r, atol=1e-9)
    np.testing.assert_allclose(op.h_t_v, expected_t, atol=1e-9)
    assert not np.allclose(op.h_t_v, nominal)


def test_derived_pair_maps_unity_axes_onto_robot_world_axes(bus):
    """Hand motion must command the same physical direction for any baseline."""
    axis_map = _unity_to_robot()
    for flange_angle_deg, hand_frame in [
        (0.0, np.diag([-1.0, 1.0, -1.0])),
        (35.0, np.diag([-1.0, 1.0, -1.0])),
        (35.0, _rot_x(np.deg2rad(40.0)) @ np.diag([-1.0, 1.0, -1.0])),
    ]:
        op = _make_operator(transformed_keypoints_port=5555, endeff_publish_port=10032)
        flange = _rot_x(np.deg2rad(flange_angle_deg))
        _publish_robot_pose(bus, 10031, flange, position=(0.3, -0.1, 0.2))
        _publish_hand_frame(bus, 5555, hand_frame)
        op._apply_retargeted_angles()
        baseline = bus.recv_latest(10032, "endeff_coords")
        assert baseline is not None

        for axis in range(3):
            offset = np.zeros(3)
            offset[axis] = 0.01
            _publish_hand_frame(bus, 5555, hand_frame, origin=offset)
            op._apply_retargeted_angles()
            cmd = bus.recv_latest(10032, "endeff_coords")
            delta = np.asarray(cmd.position_m) - np.asarray(baseline.position_m)
            np.testing.assert_allclose(delta, axis_map @ offset, atol=1e-6)

        # The reset must not ask the arm to rotate away from where it is.
        quat = np.asarray(baseline.orientation_xyzw)
        from scipy.spatial.transform import Rotation

        np.testing.assert_allclose(
            Rotation.from_quat(quat).as_matrix(), flange, atol=1e-6
        )


def _random_rotation(rng):
    rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    return rotation


def test_calibration_pair_satisfies_the_pipeline_algebra():
    """Pure check: the solved pair reproduces the axis map and holds the baseline."""
    axis_map = _unity_to_robot()
    rng = np.random.default_rng(7)
    for _ in range(20):
        r_init = _random_rotation(rng)
        r_hand = _random_rotation(rng)

        h_r_v, h_t_v = OneroOperator._calibration_from_baselines(
            np.block([[r_init, np.zeros((3, 1))], [0, 0, 0, 1]]),
            np.block([[r_hand, np.zeros((3, 1))], [0, 0, 0, 1]]),
        )

        # translation: d_cmd = R_init @ inv(H_T_V) @ R_hand^T @ d
        translation_map = r_init @ h_t_v[:3, :3].T @ r_hand.T
        np.testing.assert_allclose(translation_map, axis_map, atol=1e-9)

        # rotation: R_cmd = R_init @ inv(H_R_V) @ R_rel @ H_R_V
        # The world-frame rotation lands on the outside, i.e. it is applied to the
        # orientation the arm already had, not about the tool's own axes.
        r_rel = _random_rotation(rng)
        r_cmd = r_init @ h_r_v[:3, :3].T @ r_rel @ h_r_v[:3, :3]
        r_u = r_hand @ r_rel @ r_hand.T
        np.testing.assert_allclose(r_cmd, axis_map @ r_u @ axis_map.T @ r_init, atol=1e-9)
