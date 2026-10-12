"""Registry-level regression guard for adding a new robot.

These checks are deliberately dependency-light: they must pass on a machine
that only installed the base requirements, because the whole point of the lazy
``onero_sdk`` import is that selecting xArm7 never needs the Onero SDK.

They also encode the port-allocation rule that the left arm must offset its
ports far enough not to land on the right arm's allocation -- a mistake that is
invisible to a naive "no duplicate within one column" assertion.
"""

import importlib

from beavr.teleop.common.configs.loader import Laterality, load_robot_config
from beavr.teleop.configs.constants import ports, robots
from beavr.teleop.configs.robots import TeleopRobotConfig

# The registry is populated by import side effects, so every machine under test
# has to be imported before the lookup assertions below.
importlib.import_module("beavr.teleop.configs.robots.onero_config")
importlib.import_module("beavr.teleop.configs.robots.xarm7_config")


def _allocation(robot_cfg) -> dict:
    return {
        "endeff_publish": robot_cfg.endeff_publish_port,
        "endeff_subscribe": robot_cfg.endeff_subscribe_port,
        "joint_subscribe": robot_cfg.joint_subscribe_port,
        "reset_subscribe": robot_cfg.reset_subscribe_port,
        "state_publish": robot_cfg.state_publish_port,
    }


def test_onero_registration_does_not_disturb_xarm7():
    """Registering a subclass must not shadow an existing choice."""
    onero = TeleopRobotConfig.get_choice_class(robots.ROBOT_NAME_ONERO)
    xarm = TeleopRobotConfig.get_choice_class(robots.ROBOT_NAME_XARM7)

    assert onero is not None
    assert xarm is not None
    assert onero is not xarm


def test_xarm7_config_still_loads_after_adding_onero():
    """A previous machine must keep building untouched."""
    config = load_robot_config(robots.ROBOT_NAME_XARM7, Laterality.RIGHT)

    assert config.robot_name == robots.ROBOT_NAME_XARM7
    assert config.robots
    assert config.operators


def test_onero_ports_never_collide_with_another_robots_ports():
    """The Onero block must sit outside every other robot allocation."""
    onero_values = {
        value
        for name, value in vars(ports).items()
        if name.startswith("ONERO_") and name.endswith("_PORT") and isinstance(value, int)
    }
    other_values = {
        value
        for name, value in vars(ports).items()
        if not name.startswith("ONERO_") and name.endswith("_PORT") and isinstance(value, int)
    }

    assert onero_values, "no ONERO_*_PORT constants found"
    assert not (onero_values & other_values), (
        f"Onero port clashes with an existing robot: {sorted(onero_values & other_values)}"
    )


def test_bimanual_left_arm_does_not_land_on_the_right_arm_ports():
    """Regression: a +1 left offset makes the left publish collide with the
    right subscribe, because publish and subscribe are only one apart."""
    config = load_robot_config(robots.ROBOT_NAME_ONERO, Laterality.BIMANUAL)
    right, left = config.robots

    right_ports = set(_allocation(right).values())
    left_ports = set(_allocation(left).values())

    # ``home`` and the global teleop-state port are intentionally shared, the
    # same way the xArm work-cell shares its home/pause socket across arms.
    shared_on_purpose = {
        ports.ONERO_HOME_SUBSCRIBE_PORT,
        ports.ONERO_TELEOPERATION_STATE_PORT,
    }
    overlap = (right_ports & left_ports) - shared_on_purpose

    assert not overlap, f"left arm collides with the right arm on ports {sorted(overlap)}"


def test_operator_and_robot_ports_mirror_each_other():
    """The operator talks to the robot, so each port must have a counterpart."""
    config = load_robot_config(robots.ROBOT_NAME_ONERO, Laterality.BIMANUAL)

    for robot_cfg, operator_cfg in zip(config.robots, config.operators, strict=True):
        # The operator publishes what the robot subscribes to, and vice versa.
        assert operator_cfg.endeff_publish_port == robot_cfg.endeff_subscribe_port
        assert operator_cfg.endeff_subscribe_port == robot_cfg.endeff_publish_port
