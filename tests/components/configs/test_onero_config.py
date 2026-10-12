"""Tests for the Onero A1 teleop configuration registration."""

import importlib

import pytest

from beavr.teleop.common.configs.loader import Laterality, load_robot_config
from beavr.teleop.configs.constants import network, robots
from beavr.teleop.configs.robots import TeleopRobotConfig, onero_config

# Importing the module registers the "onero" choice in the config registry.
from beavr.teleop.configs.robots.onero_config import OneroA1Config  # noqa: F401


def test_onero_is_registered_in_choice_registry():
    assert TeleopRobotConfig.get_choice_class(robots.ROBOT_NAME_ONERO) is OneroA1Config


def test_right_laterality_builds_one_arm():
    config = load_robot_config(robots.ROBOT_NAME_ONERO, Laterality.RIGHT)

    assert config.robot_name == robots.ROBOT_NAME_ONERO
    assert len(config.robots) == 1
    assert len(config.operators) == 1
    assert len(config.detector) == 1
    assert config.robots[0].is_right_arm is True
    assert config.robots[0].hand_side == robots.RIGHT


def test_left_laterality_builds_one_left_arm():
    config = load_robot_config(robots.ROBOT_NAME_ONERO, Laterality.LEFT)

    assert len(config.robots) == 1
    assert config.robots[0].is_right_arm is False
    assert config.robots[0].hand_side == robots.LEFT


def test_bimanual_laterality_builds_both_arms():
    config = load_robot_config(robots.ROBOT_NAME_ONERO, Laterality.BIMANUAL)

    assert len(config.robots) == 2
    assert len(config.operators) == 2
    assert len(config.transforms) == 2
    assert {robot.hand_side for robot in config.robots} == {robots.LEFT, robots.RIGHT}


def test_left_and_right_arms_use_distinct_ports():
    config = load_robot_config(robots.ROBOT_NAME_ONERO, Laterality.BIMANUAL)

    publish_ports = [robot.endeff_publish_port for robot in config.robots]
    subscribe_ports = [robot.endeff_subscribe_port for robot in config.robots]

    assert len(set(publish_ports)) == 2
    assert len(set(subscribe_ports)) == 2


def test_config_build_is_deferred_to_hardware_construction():
    """Building the config must not require the onero SDK or a CAN interface."""
    config = load_robot_config(robots.ROBOT_NAME_ONERO, Laterality.RIGHT)

    robot_cfg = config.robots[0]
    assert robot_cfg.can_channel == robots.ONERO_DEFAULT_CAN_CHANNEL
    assert robot_cfg.model == robots.ONERO_DEFAULT_MODEL
    assert robot_cfg.with_gripper is robots.ONERO_DEFAULT_WITH_GRIPPER
    # None means "keep the SDK default" until the gains are tuned on hardware.
    assert robot_cfg.kp is None
    assert robot_cfg.kd is None
    assert robot_cfg.recorder_config["robot_identifier"] == robots.ROBOT_IDENTIFIER_RIGHT_ONERO


def test_mounting_preset_is_explicit_and_mirrored():
    """``None`` silently becomes UPRIGHT, which rotates gravity by 90 degrees.

    The A1 is mounted sideways -- the SDK's own mounting.py says so -- so the
    preset has to be stated explicitly, and the two arms have to be mirrored.
    Getting this wrong aims the gravity feedforward sideways instead of up:
    the joints lag under load, and a servo hold against a rigid support becomes
    a sustained stall (screech and heat) instead of tracking.
    """
    assert robots.ONERO_DEFAULT_MOUNTING == "humanoid_right"
    assert robots.ONERO_LEFT_DEFAULT_MOUNTING == "humanoid_left"

    right = load_robot_config(robots.ROBOT_NAME_ONERO, Laterality.RIGHT)
    left = load_robot_config(robots.ROBOT_NAME_ONERO, Laterality.LEFT)

    assert right.robots[0].mounting == "humanoid_right"
    assert left.robots[0].mounting == "humanoid_left"


def test_no_arm_falls_back_to_the_none_mounting():
    """A ``None`` mounting is the exact footgun this guards against."""
    config = load_robot_config(robots.ROBOT_NAME_ONERO, Laterality.BIMANUAL)

    for robot_cfg in config.robots:
        assert robot_cfg.mounting is not None
        assert robot_cfg.mounting.startswith("humanoid_")


def test_can_channel_defaults_to_the_native_socketcan_interface():
    config = load_robot_config(robots.ROBOT_NAME_ONERO, Laterality.RIGHT)

    assert config.robots[0].can_channel == robots.ONERO_DEFAULT_CAN_CHANNEL
    assert config.robots[0].can_channel == "can0"


def test_vr_freq_defaults_to_the_shipped_value():
    """VR_FREQ is the servo rate for the whole pipeline, not just the headset."""
    assert robots.VR_FREQ == 30


def test_servo_gains_default_to_inheriting_the_sdk():
    """None means "leave the SDK alone", which is the validated configuration.

    Resist adding a tuning path back. Measured on this bench at 100 Hz, every
    flat pair is wrong somewhere, because servoj broadcasts one scalar while the
    model's calibrated gains are per-joint (kp 150,150,150,150,40,40,40 and kd
    4,4,4,4,1,1,1):

      kp=50,  kd=1 -> proximal joints sit below their friction breakaway band
                      (1.5 Nm / 50 = 1.72 deg on J1), so only 2 of 7 joints
                      tracked and the Cartesian error held at 11.8 mm;
      kp=150, kd=1 -> 5 of 7 joints tracked and the error fell to 4.8 mm, but
                      J1-J4 are under-damped against their calibrated kd of 4.0
                      and the arm oscillated audibly;
      kp=150, kd=4 -> the wrist gets 4x its calibrated damping of 1.0 and
                      screeches instead.
    """
    config = load_robot_config(robots.ROBOT_NAME_ONERO, Laterality.RIGHT)

    assert config.robots[0].kp is None
    assert config.robots[0].kd is None


def test_park_pose_defaults_to_the_derived_pose():
    """home() goes here, not to zero, which is 247 mm below the table."""
    config = load_robot_config(robots.ROBOT_NAME_ONERO, Laterality.RIGHT)

    assert config.robots[0].park_pose_deg == robots.ONERO_SAFE_PARK_POSE_DEG


def test_park_pose_must_cover_every_joint():
    """A short pose would otherwise surface far later, as a refused home."""
    with pytest.raises(ValueError, match="park_pose_deg needs"):
        onero_config.OneroRobotCfg(park_pose_deg=[0.0, 0.0])


def test_park_pose_is_overridable_from_the_environment(monkeypatch):
    """The pose belongs to the cell, and no CLI route reaches it."""
    monkeypatch.setenv("ONERO_PARK_POSE_DEG", "1,2,3,4,5,6,7")
    try:
        importlib.reload(robots)

        assert robots.ONERO_SAFE_PARK_POSE_DEG == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    finally:
        monkeypatch.delenv("ONERO_PARK_POSE_DEG")
        importlib.reload(robots)


def test_a_malformed_park_pose_override_keeps_the_default(monkeypatch):
    """A typo in a launch script must not stop teleop from starting."""
    monkeypatch.setenv("ONERO_PARK_POSE_DEG", "17.14,oops")
    try:
        importlib.reload(robots)

        assert len(robots.ONERO_SAFE_PARK_POSE_DEG) == robots.ONERO_JOINT_COUNT
    finally:
        monkeypatch.delenv("ONERO_PARK_POSE_DEG")
        importlib.reload(robots)


def test_a_blank_can_channel_override_means_unset(monkeypatch):
    """A blank interface name would reach the adapter and fail misleadingly."""
    monkeypatch.setenv("ONERO_CAN_CHANNEL", "   ")
    try:
        importlib.reload(robots)

        assert robots.ONERO_DEFAULT_CAN_CHANNEL == "can0"
    finally:
        monkeypatch.delenv("ONERO_CAN_CHANNEL")
        importlib.reload(robots)


def test_can_channel_is_overridable_from_the_environment(monkeypatch):
    """The interface name belongs to the rig, not to this code.

    OneroA1Config has no channel field of its own and load_robot_config() takes
    only robot_name and laterality, so the environment is the one override
    point. A rig wired through a serial CAN bridge (slcand -> "slcan0") cannot
    start the robot without it: the adapter tries can0 and _create_arm dies with
    an OneroConnectionError.
    """
    monkeypatch.setenv("ONERO_CAN_CHANNEL", "slcan0")
    try:
        importlib.reload(robots)

        assert robots.ONERO_DEFAULT_CAN_CHANNEL == "slcan0"
    finally:
        monkeypatch.delenv("ONERO_CAN_CHANNEL")
        importlib.reload(robots)


def test_bimanual_arms_must_not_share_one_bus(monkeypatch):
    """Two arms on one channel would interleave frames on the same wire."""
    monkeypatch.setattr(robots, "ONERO_DEFAULT_CAN_CHANNEL", "slcan0")
    monkeypatch.setattr(robots, "ONERO_LEFT_CAN_CHANNEL", "slcan0")

    with pytest.raises(ValueError, match="separate buses"):
        OneroA1Config(laterality=Laterality.BIMANUAL)


def test_components_read_the_host_address_at_build_time(monkeypatch):
    """The host must be read when the config is built, not frozen at import.

    _configure_for_laterality() passes host=network.HOST_ADDRESS to every
    component config, so patching the constant has to move all of them.
    """
    monkeypatch.setattr(network, "HOST_ADDRESS", "127.0.0.1")

    config = OneroA1Config(laterality=Laterality.RIGHT)

    assert config.robots[0].host == "127.0.0.1"
    assert config.operators[0].host == "127.0.0.1"
