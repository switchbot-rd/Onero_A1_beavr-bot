"""Tests for the Onero A1 teleoperation operator."""

import time

import numpy as np
import pytest

from beavr.teleop.components.detector.detector_types import InputFrame, SessionCommand
from beavr.teleop.components.interface.interface_types import CartesianState
from beavr.teleop.components.operator.operator_types import CartesianTarget
from beavr.teleop.components.operator.robots.onero_operator import OneroOperator
from beavr.teleop.configs.constants import robots


def _make_operator(
    hand_side=robots.RIGHT,
    transformed_keypoints_port=5555,
    endeff_publish_port=10032,
    teleoperation_state_port=None,
):
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
        teleoperation_state_port=teleoperation_state_port,
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


# -----------------------------------------------------------------------------
# Gripper trigger (thumb/pinky pinch -> two-state command)
# -----------------------------------------------------------------------------
def _hand(span_m: float) -> np.ndarray:
    """Keypoints with the thumb and pinky tips ``span_m`` apart.

    Both tips sit away from the origin so they are not mistaken for the untracked
    (all-zero) joints the App streams when tracking drops.
    """
    keypoints = np.zeros((robots.OCULUS_NUM_KEYPOINTS, 3))
    keypoints[OneroOperator._THUMB_TIP_IDX] = (0.0, 0.0, 0.40)
    keypoints[OneroOperator._PINKY_TIP_IDX] = (span_m, 0.0, 0.40)
    return keypoints


def _feed(op, spans):
    """Run samples through the classifier, applying the state like the loop does."""
    decisions = []
    for span in spans:
        state = op._classify_pinch(span)
        if state is not None:
            op._gripper_is_closed = state
        decisions.append(state)
    return decisions


def test_pinch_distance_measures_the_tip_span():
    op = _make_operator()

    assert op._pinch_distance(_hand(0.04)) == pytest.approx(0.04)


def test_pinch_distance_rejects_a_frame_with_a_dropped_tip():
    """A dropped joint streams as the origin; the span would read as 0.4 m.

    That bogus "fingers far apart" reading is what opened the gripper mid-pinch
    whenever headset tracking lost a fingertip while the hand was moving.
    """
    op = _make_operator()
    keypoints = _hand(0.02)
    keypoints[OneroOperator._PINKY_TIP_IDX] = (0.0, 0.0, 0.0)

    assert op._pinch_distance(keypoints) is None


def test_pinch_distance_rejects_an_untracked_hand():
    op = _make_operator()

    assert op._pinch_distance(np.zeros((robots.OCULUS_NUM_KEYPOINTS, 3))) is None


def test_gripper_ignores_a_single_wide_frame_while_closed():
    """One jittery frame must not drop what the gripper is holding."""
    op = _make_operator()
    op._gripper_is_closed = True

    decisions = _feed(op, [0.02, 0.02, 0.02, 0.09, 0.02, 0.02])

    assert decisions == [None] * 6
    assert op._gripper_is_closed is True


def test_gripper_opens_after_a_sustained_spread():
    op = _make_operator()
    op._gripper_is_closed = True
    _feed(op, [0.02, 0.02, 0.02])

    decisions = _feed(op, [0.09] * 5)

    assert decisions[0] is None, "one wide frame is not enough"
    assert next(d for d in decisions if d is not None) is False
    assert op._gripper_is_closed is False


def test_gripper_closes_after_a_sustained_pinch():
    op = _make_operator()

    decisions = _feed(op, [0.02] * 4)

    assert next(d for d in decisions if d is not None) is True
    assert op._gripper_is_closed is True


class _StubSubscriber:
    """Serves one fixed keypoint frame, like the cached detector subscriber."""

    def __init__(self, frame):
        self._frame = frame

    def recv_keypoints(self):
        return self._frame


class _RecordingPublisher:
    def __init__(self):
        self.commands = []

    def publish(self, **kwargs):
        self.commands.append(kwargs["data"].command)


def test_update_gripper_publishes_only_after_the_debounce():
    """End to end through ``_update_gripper``: jitter must send nothing."""
    op = _make_operator()
    publisher = _RecordingPublisher()
    op._gripper_publisher = publisher
    op._gripper_command_port = 10040
    op._gripper_is_closed = True

    def feed(span, frames=1):
        op._raw_keypoints_subscriber = _StubSubscriber(
            InputFrame(
                timestamp_s=time.time(),
                hand_side=robots.RIGHT,
                keypoints=_hand(span).tolist(),
                is_relative=False,
            )
        )
        for _ in range(frames):
            op._update_gripper()

    feed(0.02, frames=3)  # sustained pinch, already closed
    feed(0.09, frames=1)  # one jittery frame
    assert publisher.commands == []

    feed(0.09, frames=3)  # sustained spread
    assert publisher.commands == ["open"]


def test_gripper_does_not_oscillate_inside_the_hysteresis_band():
    """A hand resting between the thresholds must hold its state."""
    op = _make_operator()
    op._gripper_is_closed = True
    inside = (op._PINCH_CLOSE_M + op._PINCH_OPEN_M) / 2

    decisions = _feed(op, [inside] * 10)

    assert decisions == [None] * 10
    assert op._gripper_is_closed is True


def test_gripper_closes_at_the_span_a_rotated_palm_reports():
    """Tracking over-reports the tip gap when the palm faces away from the headset.

    Real pinches were measured around 0.04 m, above the old 0.03 m close
    threshold, so those angles never closed at all.
    """
    op = _make_operator()

    decisions = _feed(op, [0.04] * 3)

    assert next(d for d in decisions if d is not None) is True
    assert op._gripper_is_closed is True


def test_gripper_stays_open_on_a_resting_hand():
    """A relaxed hand measures 0.08-0.10 m; the wider band must not close on it."""
    op = _make_operator()

    decisions = _feed(op, [0.09] * 6)

    assert decisions == [None] * 6
    assert op._gripper_is_closed is False


# ---------------------------------------------------------------------------------
# Start gate
# ---------------------------------------------------------------------------------


def _publish_pause(bus, port, command, timestamp_s):
    bus.publish(
        "127.0.0.1",
        port,
        robots.PAUSE,
        SessionCommand(timestamp_s=timestamp_s, command=command),
    )


def test_start_gate_holds_until_the_app_presses_start(bus, monkeypatch):
    """The App pushes keypoints before Start; the gate must still hold the arm.

    Nothing may be published, and just as important the inherited reset must not
    run: it blocks until a hand frame arrives and rebaselines the hand-to-arm
    mapping, so running it on the menu screen both stalls the operator and
    captures a baseline nobody intended.
    """
    monkeypatch.setenv("ONERO_REQUIRE_START", "1")
    op = _make_operator(teleoperation_state_port=10050)
    assert op.arm_teleop_state == robots.ARM_TELEOP_STOP

    # Menu screen: the detector republishes the App's "Low" as "resume".
    for index in range(3):
        _publish_pause(bus, 10050, robots.RESUME, 1.0 + index)
        op._apply_retargeted_angles()

    assert op.arm_teleop_state == robots.ARM_TELEOP_STOP
    assert op.is_first_frame is True
    assert bus.recv_latest(10032, "endeff_coords") is None


def test_start_gate_opens_on_the_continue_code(bus, monkeypatch):
    """The App's "High" arrives as "pause"; that is the one code that starts teleop."""
    monkeypatch.setenv("ONERO_REQUIRE_START", "1")
    op = _make_operator(teleoperation_state_port=10050)

    flange = _rot_x(np.deg2rad(35.0))
    hand = np.diag([-1.0, 1.0, -1.0])
    _publish_robot_pose(bus, 10031, flange)
    _publish_hand_frame(bus, 5555, hand)

    _publish_pause(bus, 10050, robots.RESUME, 1.0)
    op._apply_retargeted_angles()
    assert op.is_first_frame is True
    assert bus.recv_latest(10032, "endeff_coords") is None

    _publish_pause(bus, 10050, robots.PAUSE, 2.0)
    op._apply_retargeted_angles()

    assert op.arm_teleop_state == robots.ARM_TELEOP_CONT
    assert op.is_first_frame is False
    assert bus.recv_latest(10032, "endeff_coords") is not None
