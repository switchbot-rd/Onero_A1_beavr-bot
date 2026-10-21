"""Onero A1 teleop interface for the Beavr teleoperation stack.

The interface bridges the operator's end-effector commands onto an Onero A1 arm
driven by the ``onero-sdk`` package (``pip install onero-sdk``).

Data flow (identical to the other Beavr robot interfaces):

    Operator ──"endeff_coords"──▶ OneroRobot ──CAN──▶ Onero A1 hardware
    OneroRobot ──"endeff_homo"──▶ Operator  (teleop reset baseline)

Wire format (unchanged):
- subscribes  "endeff_coords" -> CartesianTarget (position in meters, quaternion)
- subscribes  "reset"         -> SessionCommand
- subscribes  "home"          -> SessionCommand
- subscribes  "pause"         -> SessionCommand
- publishes   "endeff_homo"   -> CartesianState with a 4x4 homogeneous matrix
- publishes   <self.name>     -> state dictionary consumed by recorders

Conventions:
- Positions are exchanged in meters and orientations as XYZW quaternions.
- The onero SDK takes the same metric position plus a rotation vector, so the
  only conversion needed here is quaternion -> rotation vector.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from beavr.teleop.common.network.handshake import HandshakeCoordinator
from beavr.teleop.common.network.publisher import ZMQPublisherManager
from beavr.teleop.common.network.subscriber import ZMQSubscriber
from beavr.teleop.common.network.utils import cleanup_zmq_resources
from beavr.teleop.common.ops import Ops
from beavr.teleop.components.detector.detector_types import SessionCommand
from beavr.teleop.components.interface.interface_base import RobotWrapper
from beavr.teleop.components.interface.interface_types import (
    CartesianState,
    CommandedCartesianState,
)
from beavr.teleop.components.operator.operator_types import CartesianTarget
from beavr.teleop.configs.constants import ports, robots

try:  # onero-sdk is imported lazily elsewhere; keep module import hardware-free
    from onero_sdk.exceptions import IKConvergenceError
except Exception:  # pragma: no cover - SDK absent (tests, dry runs)

    class IKConvergenceError(Exception):
        """Placeholder so the servo guard stays valid without the SDK."""


logger = logging.getLogger(__name__)


class OneroRobot(RobotWrapper):
    """Beavr robot interface driving an Onero A1 arm through ``onero-sdk``.

    The arm is created lazily on construction. Pass ``arm=<instance>`` to inject
    an already created SDK arm (used by tests and by the SDK simulation backend).
    """

    # Seconds allowed for the park-pose move in ``home()``. Matches the SDK's own
    # ``Arm.home()`` duration (3 s) so a homing command takes the same wall-clock
    # time it always did, just to a pose that is clear of the table.
    _PARK_MOVE_DURATION_S = 3.0

    def __init__(
        self,
        host: str,
        endeff_subscribe_port: int,
        joint_subscribe_port: int,
        home_subscribe_port: int,
        reset_subscribe_port: int,
        teleoperation_state_port: int,
        can_channel: str = "can0",
        model: str = robots.ONERO_DEFAULT_MODEL,
        mounting: Optional[str] = None,
        world_frame: bool = True,
        with_gripper: bool = False,
        gripper_enabled: bool = False,
        gripper_command_port: Optional[int] = None,
        is_right_arm: bool = True,
        endeff_publish_port: int = ports.ONERO_ENDEFF_PUBLISH_PORT,
        state_publish_port: int = ports.ONERO_STATE_PUBLISH_PORT,
        command_frame: Optional[str] = None,
        stiffness: Optional[str] = None,
        home_on_start: bool = False,
        park_pose_deg: Optional[list[float]] = None,
        max_home_travel_deg: float = 170.0,
        max_step_distance_m: float = 0.05,
        max_step_rotation_rad: float = 0.5,
        kp: Optional[float] = None,
        kd: Optional[float] = None,
        arm: Any = None,
        **kwargs,
    ):
        """
        Args:
            host: Network host address.
            endeff_subscribe_port: Port carrying the operator's "endeff_coords" commands.
            joint_subscribe_port: Reserved for joint command streams (unused today,
                kept for parity with the other robot interfaces).
            home_subscribe_port: Port carrying the "home" command.
            reset_subscribe_port: Port carrying the "reset" command.
            teleoperation_state_port: Port carrying the "pause" command.
            can_channel: SocketCAN interface the A1 is wired to (e.g. "can0").
            model: Arm model name resolved by the SDK registry ("A1R" / "A1L").
            mounting: Installation preset forwarded to the SDK (e.g.
                "side_mount_left", "humanoid_right"). ``None`` means upright.
            world_frame: When True the SDK Cartesian APIs use the intuitive
                world frame (X=forward, Y=left, Z=up).
            with_gripper: Tell the SDK that a gripper is fitted so it applies the
                validated per-joint gravity profile. This matters: the profile
                multiplies the modelled gravity torque by up to 5.3x (J5) and
                2.8x (J3), so leaving it ``False`` on a gripper-equipped arm
                under-compensates gravity and the joints sag under load. Declare
                a measured mass/CoG with ``Arm.set_payload()`` as well when known.
            is_right_arm: Whether this interface drives the right arm.
            endeff_publish_port: Port used to publish "endeff_homo" to the operator.
            state_publish_port: Port used to publish the recorded state dictionary.
            command_frame: Reference frame for Cartesian targets and pose reads.
                ``None`` (default) defers to the SDK, which resolves it from
                ``world_frame``. Pass an explicit name only to override that.
                Do NOT hard-code "base" while also passing ``world_frame=True``:
                the SDK's ``_resolve_frame`` only uses its default when the
                caller passes ``None``, so an explicit "base" silently wins and
                ``world_frame`` becomes a no-op. The two coordinate systems
                differ by the mounting rotation (90 degrees on a side-mounted
                A1), which is exactly the class of error that makes a teleop
                arm drive the wrong way.
            home_on_start: Whether the control loop homes the arm before it
                starts tracking. ``False`` by default: ``Arm.home()`` moves every
                joint to zero, which on this cell is 247 mm *below* the table.
                See ``_startup_home``.
            stiffness: Impedance preset for the SDK's adaptive GainPolicy, which
                supplies the gains for planned motions (movej/movel/movec):
                "stiff" (omega_n=40), "medium" (20) or "soft" (8). ``None``
                inherits the SDK's own default (STIFF), which is what the
                validated teleop deployment in this repo runs. The SDK's debugui
                starts at "medium"; pass that to copy its softer feel. This does
                NOT affect servoj/servop, which use ``kp``/``kd``.
            max_step_distance_m: Hard clamp on the per-cycle translation (meters).
            max_step_rotation_rad: Hard clamp on the per-cycle rotation (radians).
            kp: Position gain forwarded to the SDK servo. ``None`` keeps the SDK
                default, which is a uniform ``50.0``. The SDK also ships
                calibrated per-joint gains ``(150, 150, 150, 150, 40, 40, 40)``,
                but its servo entry points only accept a scalar, so passing
                ``150.0`` here is the closest approximation. Measured on an A1R,
                the soft default lets joints sag 0.7-1.6 deg under gravity
                (5-25 mm at the tool, enough for the SDK to log
                "Servo tracking lag").
            kd: Damping gain forwarded to the SDK servo. ``None`` keeps the SDK
                default (``1.0``).
            arm: Optional pre-built SDK arm; when omitted the SDK creates one.
        """
        self._can_channel = can_channel
        self._model = model
        self._mounting = mounting
        self._world_frame = world_frame
        self._with_gripper = with_gripper
        # Set before _create_arm() runs: that call decides whether to attach the
        # effector object, and the SDK arm must know about it at construction time.
        self._gripper_enabled = bool(gripper_enabled)
        self._gripper_is_closed = False
        self._gripper_failures = 0
        # The A1 gripper shares the arm's CAN transport and I/O lock, and its
        # transaction waits up to the feedback timeout (50 ms) while the bus
        # reader discards every non-0x18 frame -- i.e. all seven motor feedback
        # frames. Retrying a failing command on every 100 Hz servo tick therefore
        # starves the servo of feedback and freezes the arm, which looks exactly
        # like "teleop hangs the moment I pinch". Back off between attempts.
        self._gripper_retry_after = 0.0
        self._gripper_retry_interval_s = 1.0
        self._gripper_command_port = gripper_command_port
        self._is_right_arm = is_right_arm
        self._command_frame = command_frame
        self._stiffness = stiffness
        self._home_on_start = home_on_start
        # Defaulting to the derived pose rather than to None keeps a bare
        # OneroRobot safe: None would fall back to Arm.home(), i.e. every joint
        # to zero, which is the table-inbound move this exists to avoid. Pass an
        # empty list to opt out deliberately.
        if park_pose_deg is None:
            park_pose_deg = robots.ONERO_SAFE_PARK_POSE_DEG
        self._park_pose_deg = list(park_pose_deg) or None
        self._max_home_travel_deg = float(max_home_travel_deg)
        self._max_step_distance_m = max_step_distance_m
        self._max_step_rotation_rad = max_step_rotation_rad
        self._kp = kp
        self._kd = kd
        # Consecutive IK failures. The SDK servo path raises when its internal
        # solver cannot reach the target in the allotted iterations; a single
        # unreachable frame must never take the whole component down.
        self._ik_failures = 0
        self._ik_last_log = 0.0
        self._nonfinite_commands = 0
        self._data_frequency = robots.VR_FREQ
        self._recorder_config = kwargs.get("recorder_config", {})
        self.robot_identifier = self._recorder_config.get("robot_identifier", self.name)

        # Hardware handle: injected for tests/simulation, otherwise built below.
        self._arm = arm if arm is not None else self._create_arm()

        # Subscribers -----------------------------------------------------------------
        self._cartesian_coords_subscriber = ZMQSubscriber(
            host=host,
            port=endeff_subscribe_port,
            topic="endeff_coords",
            message_type=CartesianTarget,
        )
        self._reset_subscriber = ZMQSubscriber(
            host=host,
            port=reset_subscribe_port,
            topic="reset",
            message_type=SessionCommand,
        )
        self._home_subscriber = ZMQSubscriber(
            host=host,
            port=home_subscribe_port,
            topic="home",
            message_type=SessionCommand,
        )
        self._arm_teleop_state_subscriber = Ops(
            arm_teleop_state_subscriber=ZMQSubscriber(
                host=host,
                port=teleoperation_state_port,
                topic="pause",
                message_type=SessionCommand,
            )
        )

        self._subscribers = {
            "cartesian_coords": self._cartesian_coords_subscriber,
            "reset": self._reset_subscriber,
            "home": self._home_subscriber,
            "teleop_state": self._arm_teleop_state_subscriber.get_arm_teleop_state,
        }

        # Gripper --------------------------------------------------------------------
        # Two-state only ("open" / "close"): the App never transmits a grasp
        # amount, so the operator derives the state from the thumb/index distance
        # and sends a discrete command. Nothing here is proportional on purpose.
        self._gripper_subscriber = None
        if self._gripper_enabled and gripper_command_port is not None:
            self._gripper_subscriber = ZMQSubscriber(
                host=host,
                port=gripper_command_port,
                topic="gripper",
                message_type=SessionCommand,
            )

        # Publishers ------------------------------------------------------------------
        self._publisher_manager = ZMQPublisherManager.get_instance()
        self._publisher_host = host
        self._endeff_publish_port = endeff_publish_port
        self._state_publish_port = state_publish_port

        # Caches ----------------------------------------------------------------------
        self._latest_cartesian_coords: Optional[np.ndarray] = None
        self._latest_joint_state: Optional[np.ndarray] = None
        self._latest_cartesian_state_timestamp = 0.0
        self._latest_joint_state_timestamp = 0.0
        self._latest_commanded_cartesian_position: Optional[np.ndarray] = None
        self._latest_commanded_cartesian_timestamp = 0.0
        self._is_homed = False

        # Handshake coordination between operator and robot ---------------------------
        self._handshake_coordinator = HandshakeCoordinator.get_instance()
        self._handshake_server_id = f"{self.name}_handshake"
        self._handshake_coordinator.start_server(
            subscriber_id=self._handshake_server_id,
            bind_host="*",
            port=robots.TELEOP_HANDSHAKE_PORT + (1 if self._is_right_arm else 2),
        )
        logger.info(f"Handshake server started for {self.name}")

    # ---------------------------------------------------------------------------------
    # Hardware lifecycle
    # ---------------------------------------------------------------------------------
    def _create_arm(self):
        """Create and enable the SDK arm.

        Imported lazily so that the ``onero-sdk`` dependency is only required when
        an Onero robot is actually selected.
        """
        try:
            from onero_sdk import Arm
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "The Onero A1 interface requires the 'onero-sdk' package. "
                "Install it with: pip install onero-sdk"
            ) from exc

        end_effector = None
        if self._gripper_enabled:
            # The A1 gripper shares the arm's CAN bus and answers on 0x08 (commands)
            # / 0x18 (feedback); the SDK binds it to the arm's bus when no bus is
            # passed. ``with_gripper`` alone only scales the gravity model, it does
            # NOT create a controllable effector -- that needs this object.
            from onero_sdk.grippers import OneroGripper

            end_effector = OneroGripper(model=self._model)
            logger.info(f"{self.name}: gripper enabled ({type(end_effector).__name__})")

        arm = Arm(
            self._model,
            channel=self._can_channel,
            mounting=self._mounting,
            world_frame=self._world_frame,
            with_gripper=self._with_gripper,
            end_effector=end_effector,
        )
        # Planned motions (movej/movel/movec) take their gains from the SDK's
        # adaptive GainPolicy, not from the flat kp/kd the servo path uses.
        # GainPolicy's own default is STIFF (omega_n=40). The SDK's debugui
        # starts at "medium" (omega_n=20), but the validated teleop deployment
        # in this repo (oneroh1lerobot/configs/*.json) never sets a stiffness at
        # all, so it runs STIFF. Default to None -- inherit the SDK's value --
        # so this adapter matches the validated teleop pipeline; pass "medium"
        # explicitly to copy the softer debugui feel. Setting it changes J2/J3/J4
        # planned-motion gains by 2.3x-3.8x, so it is not a cosmetic knob.
        # servoj/servop are unaffected: they use kp/kd, which stay None so they
        # resolve to the SDK's 50 / 1.0 (also the debugui defaults).
        if self._stiffness is not None and hasattr(arm, "set_stiffness"):
            arm.set_stiffness(self._stiffness)

        # ``Arm`` does not enable motors implicitly; teleoperation needs them on.
        arm.enable()
        return arm

    def _safe_stop(self):
        """Best-effort stop used on shutdown paths."""
        if self._arm is None:
            return
        try:
            if hasattr(self._arm, "stop"):
                self._arm.stop()
        except Exception as exc:  # pragma: no cover - hardware dependent
            logger.warning(f"Failed to stop {self.name}: {exc}")

    # ---------------------------------------------------------------------------------
    # RobotWrapper API
    # ---------------------------------------------------------------------------------
    @property
    def name(self) -> str:
        return f"{robots.ROBOT_NAME_ONERO}_{'right' if self._is_right_arm else 'left'}"

    @property
    def recorder_functions(self) -> dict:
        """Mapping of recorded state keys to getter functions."""
        return {
            "joint_states": self.get_joint_state,
            "onero_cartesian_states": self.get_robot_actual_cartesian_position,
            "commanded_cartesian_state": self.get_cartesian_commanded_position,
            "joint_angles_rad": self.get_joint_position,
        }

    @property
    def data_frequency(self) -> float:
        return self._data_frequency

    def get_joint_state(self) -> Optional[dict]:
        positions = self.get_joint_position()
        if positions is None:
            return None
        self._latest_joint_state = np.asarray(positions, dtype=np.float32)
        self._latest_joint_state_timestamp = time.time()
        return {
            "joint_position": self._latest_joint_state.tolist(),
            "timestamp": self._latest_joint_state_timestamp,
        }

    def get_joint_position(self) -> Optional[list]:
        state = getattr(self._arm, "state", None)
        joints = getattr(state, "joints", None)
        if joints is None:
            return None
        return [float(value) for value in joints]

    def get_joint_velocity(self) -> Optional[list]:
        state = getattr(self._arm, "state", None)
        velocities = getattr(state, "velocities", None)
        if velocities is None:
            return None
        return [float(value) for value in velocities]

    def get_joint_torque(self) -> Optional[list]:
        state = getattr(self._arm, "state", None)
        torques = getattr(state, "torques", None)
        if torques is None:
            return None
        return [float(value) for value in torques]

    def get_cartesian_position(self) -> Optional[list]:
        """Current tool pose as [x, y, z (m), rx, ry, rz (rad)]."""
        position, rotation = self._arm.get_pose(frame=self._command_frame)
        rotvec = Rotation.from_matrix(np.asarray(rotation, dtype=float)).as_rotvec()
        return [float(value) for value in position] + [float(value) for value in rotvec]

    def get_pose(self) -> np.ndarray:
        """Current tool pose as a 4x4 homogeneous matrix (translation in meters)."""
        position, rotation = self._arm.get_pose(frame=self._command_frame)
        pose = np.eye(4)
        pose[:3, :3] = np.asarray(rotation, dtype=float)
        pose[:3, 3] = np.asarray(position, dtype=float)
        return pose

    def home(self):
        """Move the arm to its configured park pose.

        Deliberately *not* ``Arm.home()``, which drives every joint to zero. On
        the cell this adapter was developed against the all-zero pose puts the
        lowest link 247 mm below the plane the arm rests on, and the straight
        line there dips under that plane after roughly 5% of a 3 s move. The
        control loop calls this whenever the operator requests home, so routing
        it to zero made "home" a motion into the table.

        ``park_pose_deg`` is the cell's resting pose, derived so it clears the
        table and costs little travel from a resting arm. It is a configuration
        value because it depends on the cell, not on the arm model.

        Returns None, moving nothing, if the requested travel is absurd -- over
        170 deg on some joint, which is what a slipped decimal point looks like
        (70.22 typed as 7022) -- rather than acting on it. The limit sits far
        above any real homing move on purpose: during teleoperation the arm
        routinely rests tens of degrees from its park pose, and refusing those
        would break homing outright. This catches typos, not workspaces.
        """
        if not self._park_pose_deg:
            logger.warning(
                "No park_pose_deg configured; falling back to Arm.home(), which "
                "drives every joint to zero. That is only safe on a cell where "
                "the all-zero pose is clear of the table and any fixtures."
            )
            return self._arm.home()

        target = [float(np.deg2rad(angle)) for angle in self._park_pose_deg]
        current = self.get_joint_position()
        if current is not None:
            travel_deg = float(
                np.max(
                    np.abs(
                        np.rad2deg(np.asarray(target, dtype=float))
                        - np.rad2deg(np.asarray(current, dtype=float))
                    )
                )
            )
            if travel_deg > self._max_home_travel_deg:
                logger.error(
                    "Refusing to home: the park pose would move a joint by %.1f "
                    "deg, above the %.1f deg sanity limit. Check that "
                    "park_pose_deg is in degrees and belongs to this cell.",
                    travel_deg,
                    self._max_home_travel_deg,
                )
                return None

        logger.info(
            "Homing to the configured park pose: %s deg",
            [round(angle, 2) for angle in self._park_pose_deg],
        )
        return self._arm.movej(target, duration=self._PARK_MOVE_DURATION_S)

    def move(self, input_angles):
        """Joint-space move (blocking)."""
        return self._arm.movej([float(value) for value in input_angles])

    def move_coords(self, cartesian_coords, duration: float = 3.0):
        """Cartesian move command.

        Args:
            cartesian_coords: Either a 7-D pose [x, y, z, qx, qy, qz, qw] in
                meters, or a ``CartesianTarget`` message.
            duration: Reserved for blocking moves; real-time teleoperation
                streams single frames instead.
        """
        pose7 = self._coerce_pose7(cartesian_coords)
        if pose7 is None:
            return None
        return self._servo_to_pose(pose7)

    def arm_control(self, cartesian_pose):
        """Real-time Cartesian servo command (non-blocking, single frame)."""
        return self.move_coords(cartesian_pose)

    def reset(self):
        """Re-arm the SDK: refresh state and return the arm to servo-ready."""
        state = self.get_joint_state()
        # ``home()`` is intentionally not called here; the teleop reset only
        # re-baselines the operator, it must not command a motion by itself.
        self._latest_commanded_cartesian_position = None

        # Seed the step clamp from the *measured* pose instead of leaving it
        # empty. An empty baseline disables clamping on the very first command
        # after a reset, so a large first target used to be handed straight to
        # the solver -- exactly the case where a 10-iteration IK fails and, before
        # the guard above, killed the component.
        try:
            measured = self.get_cartesian_position()
        except Exception as exc:  # state not ready yet
            logger.warning(f"{self.name}: could not read pose at reset ({exc})")
            measured = None
        if measured is None:
            self._last_commanded_position = None
            self._last_commanded_rotvec = None
        else:
            self._last_commanded_position = np.asarray(measured[:3], dtype=float)
            self._last_commanded_rotvec = np.asarray(measured[3:], dtype=float)
        return state is not None

    # ---------------------------------------------------------------------------------
    # State publication
    # ---------------------------------------------------------------------------------
    def get_cartesian_state_from_operator(self) -> Optional[CartesianState]:
        if self._latest_cartesian_coords is None:
            return None
        position = tuple(np.asarray(self._latest_cartesian_coords, dtype=np.float32).tolist()[:3])
        return CartesianState(position_m=position, timestamp_s=self._latest_cartesian_state_timestamp)

    def get_robot_actual_cartesian_position(self) -> Optional[CartesianState]:
        cartesian_state = self.get_cartesian_position()
        if cartesian_state is None:
            return None
        position = tuple(np.asarray(cartesian_state, dtype=np.float32).tolist()[:3])
        return CartesianState(position_m=position, timestamp_s=time.time())

    def get_cartesian_commanded_position(self) -> Optional[CommandedCartesianState]:
        if self._latest_commanded_cartesian_position is None:
            return None
        commanded = np.asarray(self._latest_commanded_cartesian_position, dtype=float)
        return CommandedCartesianState(
            commanded_cartesian_position=commanded.tolist(),
            timestamp_s=self._latest_commanded_cartesian_timestamp,
        )

    def send_robot_pose(self):
        """Publish the current pose as "endeff_homo" for the operator baseline."""
        try:
            pose = self.get_pose()
            h_matrix = tuple(tuple(float(value) for value in row) for row in pose)
            self._publisher_manager.publish(
                host=self._publisher_host,
                port=self._endeff_publish_port,
                topic="endeff_homo",
                data=CartesianState(timestamp_s=time.time(), h_matrix=h_matrix),
            )
        except Exception as exc:  # pragma: no cover - hardware dependent
            logger.error(f"Failed to publish robot pose for {self.name}: {exc}")

    def publish_current_state(self):
        """Publish the state dictionary consumed by recorders/observers."""
        publish_time = time.time()
        current_state_dict: dict[str, Any] = {}

        joint_states = self.get_joint_state()
        if joint_states is not None:
            current_state_dict["joint_states"] = joint_states

        operator_cart = self.get_cartesian_state_from_operator()
        if operator_cart is not None:
            current_state_dict["operator_cartesian_states"] = operator_cart.to_dict()

        robot_cart = self.get_robot_actual_cartesian_position()
        if robot_cart is not None:
            current_state_dict["onero_cartesian_states"] = robot_cart.to_dict()

        commanded_cart = self.get_cartesian_commanded_position()
        if commanded_cart is not None:
            current_state_dict["commanded_cartesian_state"] = commanded_cart.to_dict()

        joint_angles_rad = self.get_joint_position()
        if joint_angles_rad is not None:
            current_state_dict["joint_angles_rad"] = joint_angles_rad

        current_state_dict["timestamp"] = publish_time

        self._publisher_manager.publish(
            host=self._publisher_host,
            port=self._state_publish_port,
            topic=self.name,
            data=current_state_dict,
        )

    # ---------------------------------------------------------------------------------
    # Command handling
    # ---------------------------------------------------------------------------------
    def check_reset(self) -> bool:
        return self._reset_subscriber.recv_keypoints() is not None

    def check_home(self) -> bool:
        home_signal = self._home_subscriber.recv_keypoints()
        if home_signal is None:
            return False
        command = getattr(home_signal, "command", home_signal)
        return command == robots.ARM_TELEOP_STOP

    def get_teleop_state(self) -> int:
        """Whether the operator is currently running (CONT) or paused (STOP)."""
        return self._arm_teleop_state_subscriber.get_arm_teleop_state()

    def _startup_home(self) -> None:
        """Home once at the top of the control loop, only if explicitly enabled.

        Off by default. ``home()`` now travels to ``park_pose_deg`` rather than
        to zero, so enabling this is no longer the table-inbound move it used to
        be -- but it still commands a motion nobody asked for at start-up, and
        the teleoperation reference is taken from the measured pose anyway.

        Nothing else in the ecosystem does this: the validated deployment for
        this arm (oneroh1lerobot) never moves hardware on startup, and UR,
        Franka, xArm, ALOHA and LeRobot all adopt the measured pose as the
        tracking reference instead of travelling to a stored one.
        """
        if not self._home_on_start:
            return
        logger.warning(
            "home_on_start is enabled: driving the arm to its configured park "
            "pose before tracking. Confirm that pose suits this cell."
        )
        self.home()

    def _update_gripper(self):
        """Apply the latest two-state gripper command, if one changed.

        Deliberately isolated from the servo path: a gripper that does not answer
        (unpowered, wrong CAN id, shared-bus contention) must never take the
        tracking loop down with it.
        """
        if self._gripper_subscriber is None:
            return

        # Read once: ``recv_keypoints`` hands back the cached latest frame and
        # never clears it, so a "loop until None" never terminates once traffic
        # exists -- that would stall the servo loop exactly like it stalled the
        # operator loop.
        msg = self._gripper_subscriber.recv_keypoints()
        command = getattr(msg, "command", None) if msg is not None else None
        if command not in ("open", "close"):
            return

        want_closed = command == "close"
        if want_closed == self._gripper_is_closed:
            return

        # Back-off window: see the note in __init__. Without it every servo tick
        # retries a failing exchange and eats the arm's feedback frames.
        now = time.time()
        if now < self._gripper_retry_after:
            return

        gripper = getattr(self._arm, "gripper", None)
        if gripper is None:
            if self._gripper_failures == 0:
                logger.warning(
                    f"{self.name}: gripper command received but no effector is attached"
                )
            self._gripper_failures += 1
            self._gripper_retry_after = now + self._gripper_retry_interval_s
            return

        started = time.monotonic()
        try:
            if want_closed:
                gripper.grasp()
            else:
                gripper.open()
        except Exception as exc:
            elapsed_ms = (time.monotonic() - started) * 1000.0
            self._gripper_failures += 1
            self._gripper_retry_after = time.time() + self._gripper_retry_interval_s
            if self._gripper_failures == 1 or self._gripper_failures % 20 == 0:
                logger.warning(
                    f"{self.name}: gripper {command} failed after {elapsed_ms:.1f} ms "
                    f"({self._gripper_failures}x, retrying every "
                    f"{self._gripper_retry_interval_s:.0f}s): "
                    f"{type(exc).__name__}: {exc}"
                )
            return

        elapsed_ms = (time.monotonic() - started) * 1000.0
        self._gripper_failures = 0
        self._gripper_retry_after = 0.0
        self._gripper_is_closed = want_closed
        logger.info(
            f"{self.name}: gripper {'closed' if want_closed else 'open'} "
            f"({elapsed_ms:.1f} ms)"
        )
        if elapsed_ms > 20.0:
            # Anything past a couple of servo ticks means the exchange stalled on a
            # missing reply and the arm lost feedback meanwhile.
            logger.warning(
                f"{self.name}: gripper transaction held the shared CAN bus for "
                f"{elapsed_ms:.1f} ms -- expect a visible tracking hitch"
            )

    def stream(self):
        """Main control loop: consume operator commands and drive the arm."""
        self._startup_home()
        self.send_robot_pose()

        target_interval = 1.0 / self._data_frequency
        next_frame_time = time.time()

        while True:
            current_time = time.time()
            if current_time < next_frame_time:
                time.sleep(min(next_frame_time - current_time, target_interval))
                continue
            next_frame_time = current_time + target_interval

            if self.check_home():
                if not self._is_homed:
                    self.home()
                    self._is_homed = True
                    self.send_robot_pose()
            else:
                self._is_homed = False

            if self.check_reset():
                self.send_robot_pose()

            self._update_gripper()

            if self.get_teleop_state() == robots.ARM_TELEOP_STOP:
                # Paused: keep publishing state so observers stay in sync.
                self.publish_current_state()
                continue

            msg = self._cartesian_coords_subscriber.recv_keypoints()
            if msg is not None:
                position = np.asarray(msg.position_m, dtype=np.float32)
                orientation = np.asarray(msg.orientation_xyzw, dtype=np.float32)
                # The upstream chain yields all-NaN commands whenever the headset
                # streams untracked (all-zero) joints: the transform normalises a
                # zero vector and the operator propagates the NaN through. Feeding
                # that to the solver fails on every tick and used to spray one
                # "out of reach" warning per second forever, which is misleading.
                # Drop such frames here instead of storing them as the last target.
                if not (np.isfinite(position).all() and np.isfinite(orientation).all()):
                    self._nonfinite_commands += 1
                    if self._nonfinite_commands == 1 or self._nonfinite_commands % 300 == 0:
                        logger.warning(
                            f"{self.name}: dropping non-finite target "
                            f"({self._nonfinite_commands} so far). Headset is "
                            "likely streaming untracked hands -- check the pinch/"
                            "hand tracking, this is not an out-of-reach target."
                        )
                else:
                    self._nonfinite_commands = 0
                    self._latest_commanded_cartesian_position = np.concatenate([position, orientation])
                    self._latest_commanded_cartesian_timestamp = msg.timestamp_s
                    self._latest_cartesian_coords = position.copy()
                    self._latest_cartesian_state_timestamp = msg.timestamp_s

            if self._latest_commanded_cartesian_position is not None:
                self.move_coords(self._latest_commanded_cartesian_position)

            self.publish_current_state()

            sleep_time = max(0.0, next_frame_time - time.time())
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ---------------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------------
    def _servo_to_pose(self, pose7: np.ndarray) -> Any:
        """Send a single Cartesian target to the SDK servo interface."""
        pose6 = self._pose7_to_pose6(pose7)
        position = pose6[:3]
        rotvec = pose6[3:]

        # Last line of defence: a NaN/Inf target can never be solved, and passing
        # one to the SDK produces the misleading "out of reach" warning on every
        # tick. Refuse it before clamping (comparisons against NaN are all False).
        if not (np.isfinite(position).all() and np.isfinite(rotvec).all()):
            return None

        # Per-cycle clamping keeps a dropped/glitched VR frame from producing a
        # large jerky motion. The SDK applies its own joint-space safety limits
        # on top of this.
        clamped_position, clamped_rotvec = self._clamp_step(position, rotvec)
        target = np.concatenate([clamped_position, clamped_rotvec]).tolist()

        # A frame whose target is unreachable (hand driven into the table, past a
        # joint limit, or simply too far in one step) makes the SDK solver raise.
        # Swallow it, hold the previous target and keep the loop alive: the solver
        # restarts from the measured joints every tick, so tracking resumes by
        # itself as soon as the target is reachable again. Never publish the
        # unconverged joint solution -- it is a local optimum, not a safe pose.
        try:
            state = self._arm.servop(
                target,
                kp=self._kp,
                kd=self._kd,
                frame=self._command_frame,
            )
        except IKConvergenceError as exc:
            self._ik_failures += 1
            now = time.time()
            if now - self._ik_last_log > 1.0:  # throttle: one line per second
                self._ik_last_log = now
                logger.warning(
                    f"{self.name}: IK failed on {self._ik_failures} consecutive "
                    f"frame(s), holding last target ({exc}). Target may be out of "
                    "reach -- move the hand back into the workspace."
                )
            return None

        if self._ik_failures:
            logger.info(
                f"{self.name}: IK recovered after {self._ik_failures} failed frame(s)"
            )
            self._ik_failures = 0

        self._last_commanded_position = clamped_position
        self._last_commanded_rotvec = clamped_rotvec
        return state

    def _clamp_step(self, position: np.ndarray, rotvec: np.ndarray):
        last_position = getattr(self, "_last_commanded_position", None)
        last_rotvec = getattr(self, "_last_commanded_rotvec", None)
        if last_position is None or last_rotvec is None:
            return position, rotvec

        position_delta = position - last_position
        distance = float(np.linalg.norm(position_delta))
        if distance > self._max_step_distance_m > 0:
            logger.warning(
                f"{self.name}: clamped {distance * 1000:.1f} mm command step "
                f"to {self._max_step_distance_m * 1000:.1f} mm"
            )
            position = last_position + position_delta * (self._max_step_distance_m / distance)

        # Rotation must be compared as a *relative rotation*, not by subtracting
        # two rotation vectors. Near +-180 deg the axis of a rotation vector flips,
        # so a ~70 deg motion reads as ~290 deg and tripped this limiter on almost
        # every frame. That is the difference that made the left arm stutter while
        # the right one tracked cleanly.
        previous_rotation = Rotation.from_rotvec(last_rotvec)
        delta_rotation = Rotation.from_rotvec(rotvec) * previous_rotation.inv()
        angle = float(delta_rotation.magnitude())  # shortest angle, in [0, pi]
        if angle > self._max_step_rotation_rad > 0:
            logger.warning(
                f"{self.name}: clamped {np.rad2deg(angle):.1f} deg command step "
                f"to {np.rad2deg(self._max_step_rotation_rad):.1f} deg"
            )
            step = Rotation.from_rotvec(
                delta_rotation.as_rotvec() * (self._max_step_rotation_rad / angle)
            )
            rotvec = (step * previous_rotation).as_rotvec()

        return position, rotvec

    @staticmethod
    def _coerce_pose7(cartesian_coords) -> Optional[np.ndarray]:
        """Normalise a 7-D pose from either an array-like or a typed message."""
        if cartesian_coords is None:
            return None
        if isinstance(cartesian_coords, CartesianTarget):
            return np.concatenate(
                [
                    np.asarray(cartesian_coords.position_m, dtype=float),
                    np.asarray(cartesian_coords.orientation_xyzw, dtype=float),
                ]
            )
        pose = np.asarray(cartesian_coords, dtype=float).reshape(-1)
        if pose.size != 7:
            logger.warning(f"Ignoring malformed Cartesian command with {pose.size} values")
            return None
        return pose

    @staticmethod
    def _pose7_to_pose6(pose7: np.ndarray) -> np.ndarray:
        """[x, y, z, qx, qy, qz, qw] -> [x, y, z, rx, ry, rz] (meters, radians)."""
        position = np.asarray(pose7[:3], dtype=float)
        quat = np.asarray(pose7[3:7], dtype=float)
        norm = float(np.linalg.norm(quat))
        if norm < 1e-8:
            quat = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            quat = quat / norm
        rotvec = Rotation.from_quat(quat).as_rotvec()
        return np.concatenate([position, rotvec])

    def __del__(self):
        """Stop the handshake server and release ZMQ resources."""
        if hasattr(self, "_handshake_coordinator") and hasattr(self, "_handshake_server_id"):
            try:
                self._handshake_coordinator.stop_server(self._handshake_server_id)
            except Exception as exc:  # pragma: no cover - shutdown path
                logger.warning(f"Error stopping handshake server: {exc}")
        cleanup_zmq_resources()
