"""Tests for the Onero A1 robot interface.

These tests run without hardware or the ``onero-sdk`` package: a mock arm is
injected through the ``arm=`` constructor argument.
"""

import sys
import types
from dataclasses import dataclass, field

import numpy as np
import pytest

from beavr.teleop.components.interface.interface_types import CartesianState
from beavr.teleop.components.interface.robots.onero_robot import OneroRobot
from beavr.teleop.configs.constants import robots


@dataclass
class MockArmState:
    joints: list = field(default_factory=lambda: [0.1] * robots.ONERO_JOINT_COUNT)
    velocities: list = field(default_factory=lambda: [0.0] * robots.ONERO_JOINT_COUNT)
    torques: list = field(default_factory=lambda: [0.0] * robots.ONERO_JOINT_COUNT)


class MockArm:
    """Minimal stand-in for ``onero_sdk.Arm``."""

    def __init__(self, position=(0.3, 0.0, 0.4), rotation=None):
        self.state = MockArmState()
        self._position = np.array(position, dtype=float)
        self._rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
        self.servop_calls = []
        self.movej_calls = []
        self.homed = False
        self.stopped = False

    def get_pose(self, frame=None):
        return self._position.copy(), self._rotation.copy()

    def servop(self, target, kp=None, kd=None, frame=None):
        self.servop_calls.append((np.asarray(target, dtype=float), frame))
        self.servop_gains = (kp, kd)
        return self.state

    def movej(self, joints, **kwargs):
        self.movej_calls.append(list(joints))
        return self.state

    def home(self, **kwargs):
        self.homed = True
        return self.state

    def stop(self, **kwargs):
        self.stopped = True

    def set_stiffness(self, stiffness):
        self.stiffness = stiffness

    def enable(self):  # pragma: no cover - only used by the real SDK path
        return self.state

    def disable(self):  # pragma: no cover - only used by the real SDK path
        return None


def _make_robot(arm=None, **overrides):
    kwargs = {
        "host": "127.0.0.1",
        "endeff_subscribe_port": 10032,
        "joint_subscribe_port": 10033,
        "home_subscribe_port": 10034,
        "reset_subscribe_port": 10032,
        "teleoperation_state_port": 10036,
        "endeff_publish_port": 10031,
        "state_publish_port": 10035,
        "arm": arm if arm is not None else MockArm(),
    }
    kwargs.update(overrides)
    return OneroRobot(**kwargs)


def test_name_and_frequency_reflect_arm_side():
    right = _make_robot(is_right_arm=True)
    left = _make_robot(is_right_arm=False)

    assert right.name == "onero_right"
    assert left.name == "onero_left"
    assert right.data_frequency == robots.VR_FREQ


def test_joint_and_cartesian_state_accessors():
    arm = MockArm(position=(0.3, -0.1, 0.45))
    robot = _make_robot(arm=arm)

    joint_state = robot.get_joint_state()
    assert joint_state is not None
    assert len(joint_state["joint_position"]) == robots.ONERO_JOINT_COUNT
    assert robot.get_joint_velocity() == [0.0] * robots.ONERO_JOINT_COUNT
    assert robot.get_joint_torque() == [0.0] * robots.ONERO_JOINT_COUNT

    # [x, y, z (m), rx, ry, rz (rad)]
    cartesian = robot.get_cartesian_position()
    assert len(cartesian) == 6
    np.testing.assert_allclose(cartesian[:3], [0.3, -0.1, 0.45], atol=1e-9)
    np.testing.assert_allclose(cartesian[3:], np.zeros(3), atol=1e-9)


def test_pose7_to_pose6_converts_quaternion_to_rotation_vector():
    # 90 degrees around +Z
    half = np.sqrt(0.5)
    pose6 = OneroRobot._pose7_to_pose6(np.array([0.1, 0.2, 0.3, 0.0, 0.0, half, half]))

    np.testing.assert_allclose(pose6[:3], [0.1, 0.2, 0.3], atol=1e-9)
    np.testing.assert_allclose(pose6[3:], [0.0, 0.0, np.pi / 2], atol=1e-9)


def test_move_coords_forwards_pose_to_sdk_servo():
    arm = MockArm()
    robot = _make_robot(arm=arm)

    robot.move_coords([0.3, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0])

    assert len(arm.servop_calls) == 1
    target, frame = arm.servop_calls[0]
    # None defers to the SDK, which resolves it from world_frame. Hard-coding
    # "base" here would silently override that; see
    # test_cartesian_frame_defers_to_the_sdk_default.
    assert frame is None
    np.testing.assert_allclose(target[:3], [0.3, 0.0, 0.4], atol=1e-9)
    np.testing.assert_allclose(target[3:], np.zeros(3), atol=1e-9)


def test_move_coords_clamps_large_steps():
    arm = MockArm()
    robot = _make_robot(arm=arm, max_step_distance_m=0.05)

    robot.move_coords([0.30, 0.0, 0.40, 0.0, 0.0, 0.0, 1.0])
    robot.move_coords([0.60, 0.0, 0.40, 0.0, 0.0, 0.0, 1.0])

    target, _ = arm.servop_calls[-1]
    assert target[0] == pytest.approx(0.35, abs=1e-9)


def test_move_coords_rejects_malformed_command():
    arm = MockArm()
    robot = _make_robot(arm=arm)

    robot.move_coords([0.1, 0.2, 0.3])

    assert arm.servop_calls == []


def test_move_delegates_to_sdk_movej():
    arm = MockArm()
    robot = _make_robot(arm=arm)

    robot.move([0.0] * robots.ONERO_JOINT_COUNT)

    assert arm.movej_calls == [[0.0] * robots.ONERO_JOINT_COUNT]


def test_home_goes_to_the_park_pose_not_to_zero():
    """The point of the whole change: "home" must not mean "every joint to zero".

    ``Arm.home()`` drives all seven joints to zero, and on the development cell
    that puts the lowest link 247 mm below the plane the arm rests on. The control
    loop calls ``home()`` whenever the operator asks for it, so a homing command
    used to be a command to drive into the table.
    """
    arm = MockArm()
    robot = _make_robot(arm=arm)

    robot.home()

    expected = [float(np.deg2rad(angle)) for angle in robots.ONERO_SAFE_PARK_POSE_DEG]
    assert len(arm.movej_calls) == 1
    assert np.allclose(arm.movej_calls[0], expected)
    assert arm.homed is False, "Arm.home() must not be reached"


def test_home_uses_a_configured_park_pose():
    arm = MockArm()
    robot = _make_robot(arm=arm, park_pose_deg=[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    robot.home()

    assert np.allclose(arm.movej_calls[0], [np.deg2rad(10.0)] + [0.0] * 6)


def test_home_refuses_a_park_pose_with_an_absurd_joint_value():
    """A typo turning 70.22 into 7022 must not be acted on.

    Refusing is the safe failure: a home command is operator-initiated and will be
    retried, whereas a 7000 deg command will not wait for anyone to react. The
    limit sits far above any real homing travel on purpose -- during teleop the
    arm routinely rests tens of degrees away from its park pose, and rejecting
    those would break homing outright (see the test below).
    """
    arm = MockArm()  # reports 0.1 rad (about 5.7 deg) on every joint
    robot = _make_robot(
        arm=arm, park_pose_deg=[17.14, 6.22, -5.48, 7022.0, -2.19, -18.58, 0.80]
    )

    robot.home()

    assert arm.movej_calls == []


def test_home_allows_travel_well_beyond_any_working_distance():
    """The sanity limit must not refuse real homing moves.

    During teleoperation the arm routinely sits tens of degrees away from its park
    pose, so a guard tight enough to reject those would break homing outright.
    This test is the counterweight to the one above.
    """
    arm = MockArm()  # reports 0.1 rad (about 5.7 deg) on every joint
    robot = _make_robot(arm=arm, park_pose_deg=[90.0] * robots.ONERO_JOINT_COUNT)

    robot.home()

    assert len(arm.movej_calls) == 1


def test_home_falls_back_to_sdk_home_only_when_park_pose_is_empty():
    """Opting out has to be explicit; an empty pose is the only way back to zero."""
    arm = MockArm()
    robot = _make_robot(arm=arm, park_pose_deg=[])

    robot.home()

    assert arm.homed is True
    assert arm.movej_calls == []


def test_send_robot_pose_publishes_homogeneous_matrix(bus):
    robot = _make_robot(arm=MockArm(position=(0.3, -0.1, 0.45)))

    robot.send_robot_pose()

    state = bus.recv_latest(10031, "endeff_homo")
    assert isinstance(state, CartesianState)
    assert len(state.h_matrix) == 4
    assert state.h_matrix[0][3] == pytest.approx(0.3)
    assert state.h_matrix[1][3] == pytest.approx(-0.1)
    assert state.h_matrix[2][3] == pytest.approx(0.45)


def test_publish_current_state_uses_onero_keys(bus):
    robot = _make_robot(arm=MockArm())

    robot.publish_current_state()

    state = bus.recv_latest(10035, "onero_right")
    assert state is not None
    assert "joint_states" in state
    assert "onero_cartesian_states" in state
    assert "joint_angles_rad" in state


def test_reset_clears_pending_command():
    robot = _make_robot(arm=MockArm())
    robot._latest_commanded_cartesian_position = np.zeros(7)

    assert robot.reset() is True
    assert robot._latest_commanded_cartesian_position is None


def test_configured_servo_gains_are_forwarded_to_sdk():
    """The SDK's default stiffness (kp=50) lets the A1 sag under gravity."""
    arm = MockArm()
    robot = _make_robot(arm=arm, kp=150.0, kd=4.0)

    robot.move_coords([0.3, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0])

    assert arm.servop_gains == (150.0, 4.0)


def test_servo_gains_default_to_sdk_values():
    arm = MockArm()
    robot = _make_robot(arm=arm)

    robot.move_coords([0.3, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0])

    assert arm.servop_gains == (None, None)


def test_cartesian_frame_defers_to_the_sdk_default():
    """An explicit "base" would silently beat world_frame=True.

    onero_sdk only consults its default frame when the caller passes None
    (``_resolve_frame``), so hard-coding "base" turns the SDK's ``world_frame``
    option into a no-op. The two frames differ by the mounting rotation -- 90
    degrees on a side-mounted A1 -- which is exactly how a teleop arm ends up
    being driven along the wrong axis.
    """
    arm = MockArm()
    robot = _make_robot(arm=arm)

    robot.move_coords([0.3, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0])

    assert robot._command_frame is None
    assert arm.servop_calls[-1][1] is None


def test_cartesian_frame_can_still_be_overridden():
    arm = MockArm()
    robot = _make_robot(arm=arm, command_frame="world")

    robot.move_coords([0.3, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0])

    assert arm.servop_calls[-1][1] == "world"


def test_startup_does_not_home_by_default():
    """Arm.home() sends every joint to zero, which on this cell is under the table.

    Measured at 247 mm below the plane the arm rests on, with the straight-line
    path crossing that plane after ~5% of the motion. So a homing default is not
    merely wasteful, it drives the arm into the table. The validated deployment
    for this arm (oneroh1lerobot) never moves hardware on startup either.
    """
    arm = MockArm()
    robot = _make_robot(arm=arm)

    assert robot._home_on_start is False

    robot._startup_home()

    assert arm.homed is False


def test_startup_home_runs_only_when_enabled():
    arm = MockArm()
    robot = _make_robot(arm=arm, home_on_start=True)

    robot._startup_home()

    assert len(arm.movej_calls) == 1


def test_stiffness_defaults_to_none_so_the_sdk_value_is_inherited():
    """None means "do not touch it", matching the validated teleop pipeline.

    Planned motions take their gains from the SDK's adaptive GainPolicy, whose
    own default is STIFF (omega_n=40). The SDK's debugui starts at "medium",
    but oneroh1lerobot/configs/*.json never set a stiffness, so the validated
    teleop deployment runs STIFF. Inheriting is the faithful default; forcing
    "medium" here would silently soften J2/J3/J4 planned-motion gains by
    2.3x-3.8x relative to that validated pipeline.
    """
    robot = _make_robot()

    assert robot._stiffness is None


def test_default_stiffness_leaves_the_sdk_value_untouched(monkeypatch):
    """With the default, set_stiffness must not be called at all."""
    calls = {}

    class FakeArm(MockArm):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def set_stiffness(self, stiffness):  # pragma: no cover - must not run
            calls["stiffness"] = stiffness

    monkeypatch.setitem(
        sys.modules, "onero_sdk", types.SimpleNamespace(Arm=FakeArm)
    )

    OneroRobot(
        host="127.0.0.1",
        endeff_subscribe_port=10032,
        joint_subscribe_port=10033,
        home_subscribe_port=10034,
        reset_subscribe_port=10032,
        teleoperation_state_port=10036,
        endeff_publish_port=10031,
        state_publish_port=10035,
    )

    assert "stiffness" not in calls


def test_create_arm_pushes_the_stiffness_preset_into_the_sdk(monkeypatch):
    """The preset has to reach the SDK, not just sit in an attribute."""
    calls = {}

    class FakeArm(MockArm):
        def __init__(self, *args, **kwargs):
            super().__init__()
            calls["model"] = args[0] if args else None

        def set_stiffness(self, stiffness):
            calls["stiffness"] = stiffness

        def enable(self):
            calls["enabled"] = True

    monkeypatch.setitem(
        sys.modules, "onero_sdk", types.SimpleNamespace(Arm=FakeArm)
    )

    OneroRobot(
        host="127.0.0.1",
        endeff_subscribe_port=10032,
        joint_subscribe_port=10033,
        home_subscribe_port=10034,
        reset_subscribe_port=10032,
        teleoperation_state_port=10036,
        endeff_publish_port=10031,
        state_publish_port=10035,
        stiffness="soft",
    )

    assert calls["stiffness"] == "soft"
    assert calls["enabled"] is True


def test_gripper_gravity_profile_flag_is_kept():
    """The A1 gripper profile scales the modelled gravity torque by up to 5.3x."""
    assert _make_robot(with_gripper=True)._with_gripper is True
    assert _make_robot()._with_gripper is False
