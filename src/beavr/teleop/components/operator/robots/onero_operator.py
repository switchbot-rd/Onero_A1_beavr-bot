"""Onero A1 teleoperation operator.

The retargeting pipeline (hand frame -> robot frame, filtering, publishing) is
hardware agnostic and lives in :class:`XArmOperator`. The Onero A1 operator only
supplies the calibration that maps the VR / hand-tracking frames onto the A1
frame -- and, unlike the xArm variant, it derives that calibration at teleop
reset instead of hard-coding it.

Why the calibration is derived rather than configured
-----------------------------------------------------
The inherited pipeline composes every command as::

    p_cmd = p_robot_init + R_robot_init @ inv(H_T_V)[:3,:3] @ t_rel
    R_cmd = R_robot_init @ inv(H_R_V)[:3,:3] @ R_rel @ H_R_V[:3,:3]

with ``t_rel`` the hand displacement expressed in the hand frame captured at
reset and ``R_robot_init`` the flange rotation measured at that same instant.
Both commands are therefore left-multiplied by ``R_robot_init``, so a fixed
matrix can only be right for one flange orientation: measured on this cell, the
xArm7 placeholder set sends the arm backwards when the hand moves up (max map
deviation 1.99), and even a matrix derived for the park pose collapses to 0.958
deviation when the arm happens to reset from another pose.

Solving both equations for a command that is always the fixed axis map ``W``
(Unity x right / y up / z forward -> A1 world X forward / Y left / Z up) gives
one matrix per reset, used for both constants::

    M = R_robot_init^T @ W @ R_hand_init        H_R_V = H_T_V = M^-1

Composed back through the pipeline this yields ``p_cmd = p_init + W @ d`` and
``R_cmd = W R_world W^T @ R_init``. At reset (``R_rel = I``) the rotation
collapses to ``R_robot_init`` exactly, so the first command does not ask the arm
to rotate anywhere -- which is what makes the approach safe rather than merely
pose independent. It also absorbs the hand posture at reset, so neither the
operator's hand pose nor the arm's start pose changes the direction mapping.

Verified offline against this very class by ``onero_calib_probe.py``: the
measured map equals ``W`` with max deviation 0.000 for the park pose, the
identity pose, and both with a hand frame rotated 40 degrees, while the
commanded orientation stays on the measured baseline in every case.
"""

import logging
import os
import time
from typing import Any, Dict, Optional

import numpy as np

from beavr.teleop.common.network.publisher import ZMQPublisherManager
from beavr.teleop.common.network.subscriber import ZMQSubscriber
from beavr.teleop.components.detector.detector_types import InputFrame, SessionCommand
from beavr.teleop.configs.constants import ports, robots

from .xarm7_operator import XArmOperator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------------
# Axis map
# ---------------------------------------------------------------------------------
# Unity world (what the headset streams) in A1 world axes (X forward, Y left, Z up).
# Columns are the images of the Unity basis vectors, so W @ (0, 1, 0) = (0, 0, 1):
# hand up -> arm up. Determinant is -1 because Unity is left-handed and the A1 world
# frame is right-handed; a mirror is inherent to the axis correspondence, not a bug.
_UNITY_TO_ROBOT = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)

# Hand frame the transform publishes for the nominal pinch pose (palm down,
# fingers forward, right hand), from keypoint_transform.py:
#   x = cross(palm_direction, palm_normal) = -unity x   (across the palm)
#   y = palm_normal                        = +unity y   (back of the hand)
#   z = cross(x, y)                        = -unity z   (away from the fingers)
# Reserved for the nominal pose only: the matrices actually used are recomputed
# from the measured hand frame at reset (see _calibration_from_baselines).
_NOMINAL_HAND_FRAME = np.diag([-1.0, 1.0, -1.0])


def _default_calibration(hand_frame: np.ndarray) -> np.ndarray:
    """Matrix pair for a robot that resets exactly at the identity orientation."""
    m_r = _UNITY_TO_ROBOT @ hand_frame
    h = np.eye(4)
    h[:3, :3] = m_r.T  # orthogonal matrix: transpose == inverse
    return h


# Pre-reset values. ``_reset_teleop`` replaces them with the measured pair before
# any command can be published, so these only fill the constructor argument of the
# inherited pipeline and the window between construction and the first reset.
H_R_V_RIGHT = _default_calibration(_NOMINAL_HAND_FRAME)
H_T_V_RIGHT = H_R_V_RIGHT.copy()

H_R_V_LEFT = _default_calibration(_NOMINAL_HAND_FRAME)
H_T_V_LEFT = H_R_V_LEFT.copy()


class OneroOperator(XArmOperator):
    """Operator driving an Onero A1 arm from VR hand tracking.

    Inherits the whole retargeting implementation from :class:`XArmOperator` and
    replaces only the calibration: it is re-derived at every teleop reset from the
    two baselines the reset captures, so the direction map is the fixed Unity->A1
    axis map no matter where the arm starts or how the hand happens to be held.
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
        gripper_command_port: Optional[int] = None,
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

        # Gripper ------------------------------------------------------------------
        # The headset never transmits a grasp amount: its payload is 26 joint
        # positions and the thumb/index pinch flag stays local to the App. So the
        # operator re-derives it from the raw joints on the detector's publisher
        # socket and sends a two-state command ("open"/"close").
        # Diagnostics -----------------------------------------------------------------
        self._entered = 0
        self._completed = 0
        self._debug = os.environ.get("ONERO_TELEOP_DEBUG") == "1"
        self._debug_last = 0.0
        self._kp_frames = 0

        self._gripper_command_port = gripper_command_port
        self._gripper_is_closed = False
        self._pinch_history: list[float] = []
        self._pinch_open_streak = 0
        self._pinch_close_streak = 0
        self._raw_keypoints_subscriber = None
        self._gripper_publisher = ZMQPublisherManager.get_instance()
        if gripper_command_port is not None:
            self._raw_keypoints_subscriber = ZMQSubscriber(
                host=host,
                port=ports.KEYPOINT_STREAM_PORT,
                topic=hand_side,
                message_type=InputFrame,
            )

        # Start gate ------------------------------------------------------------------
        # The inherited pipeline boots straight into ARM_TELEOP_CONT, and the App
        # starts pushing keypoints as soon as its sockets come up -- before anyone
        # presses Start. The gate reuses the pause subscriber the parent already
        # created (the onero config points it at the detector's port), boots in
        # STOP, and opens only on the App's continue code.
        #
        # OneroVRHandDetector publishes the App's inverted codes ("High" ->
        # "pause"), so the command that means *continue* is ``_start_command``.
        # Set ONERO_START_COMMAND=resume if a rig reports the opposite.
        #
        # The gate used to be opt-in: on this rig the pause channel looked dead and
        # the operator sat in STOP forever. The real cause was the detector
        # rejecting every pause payload as a malformed keypoint frame (fixed in
        # OneroVRHandDetector._receive_data). What software cannot fix is a pause
        # socket pointed at the wrong port; ``_start_fallback`` covers that case by
        # opening the gate on a keypoint-only stream once the pause channel has been
        # silent for ``_start_fallback_s``. It is off by default: the guarantee that
        # nothing moves before Start has to hold unconditionally, and a misrouted
        # pause port is a configuration fault that ``ss -tnp`` shows directly.
        self._require_start = os.environ.get("ONERO_REQUIRE_START", "1") != "0"
        self._start_command = os.environ.get("ONERO_START_COMMAND", robots.PAUSE)
        self._start_fallback = os.environ.get("ONERO_START_FALLBACK", "0") != "0"
        try:
            self._start_fallback_s = float(os.environ.get("ONERO_START_FALLBACK_S", "8"))
        except ValueError:
            # A typo in the override must not take the operator down at start-up.
            self._start_fallback_s = 8.0
        self._start_t0 = time.time()
        self._pause_last_rx = 0.0
        self._pause_last_stamp: Optional[float] = None
        self._fallback_warned = False
        # Memo for the current loop turn: the pause subscriber only reports one
        # message, so the gate decision is taken once per turn and reused by the
        # inherited pipeline instead of being re-derived from a stale cache.
        self._cycle_state: Optional[int] = None
        if self._require_start and self._arm_teleop_state_subscriber is not None:
            self.arm_teleop_state = robots.ARM_TELEOP_STOP
            logger.info(
                "[%s] start gate enabled: teleop stays stopped until the App "
                "sends %r",
                self.operator_name,
                self._start_command,
            )
        else:
            # With the gate off the pause topic must not affect the state at all, or
            # a stray pause message would stop the arm mid-run. Dropping the
            # subscriber makes the inherited lookup report CONT unconditionally.
            self._arm_teleop_state_subscriber = None

    def _get_arm_teleop_state(self) -> int:
        """Read the App's start/stop signal from the (now live) pause subscriber.

        The inherited implementation is identical apart from the code mapping: the
        detector republishes the App's inverted codes, so the command that means
        *continue* arrives as ``_start_command``.

        OneroVRHandDetector only forwards the App's two state payloads, so every
        command that arrives here is meaningful; anything else is dropped upstream
        and leaves the state untouched.
        """
        if (
            not self._require_start
            or self._arm_teleop_state_subscriber is None
        ):
            return super()._get_arm_teleop_state()

        if self._cycle_state is not None:
            return self._cycle_state

        data = self._arm_teleop_state_subscriber.recv_keypoints()
        command = getattr(data, "command", None) if data is not None else None
        state = self.arm_teleop_state
        if command is not None:
            # ``recv_keypoints`` hands back the cached latest frame, so the same
            # message is re-read on every loop: only a timestamp change is news.
            stamp = getattr(data, "timestamp_s", None)
            if stamp != self._pause_last_stamp:
                self._pause_last_stamp = stamp
                self._pause_last_rx = time.time()
                if command == self._start_command:
                    state = robots.ARM_TELEOP_CONT
                elif command in (robots.PAUSE, robots.RESUME):
                    state = robots.ARM_TELEOP_STOP

        if state == self.arm_teleop_state:
            state = self._fallback_teleop_state()

        self._cycle_state = state
        return state

    def _fallback_teleop_state(self) -> int:
        """Hold STOP unless the start channel is provably missing.

        A live pause channel always wins, including a stream of "still paused"
        codes: that is the App sitting on the menu, not a fault. The fallback only
        engages when no pause traffic has arrived for ``_start_fallback_s`` while
        hand frames keep coming -- i.e. the App is pushing but its start signal
        cannot reach this host.
        """
        if not self._start_fallback:
            return self.arm_teleop_state

        now = time.time()
        if self._pause_last_rx and now - self._pause_last_rx < self._start_fallback_s:
            return self.arm_teleop_state
        if now - self._start_t0 < self._start_fallback_s:
            return self.arm_teleop_state
        if not self._hand_stream_alive():
            return self.arm_teleop_state

        if not self._fallback_warned:
            self._fallback_warned = True
            logger.warning(
                "[%s] no pause-channel traffic for %.0fs while hand frames keep "
                "arriving: the App's start signal cannot reach this host (check the "
                "pause port mapping). Falling back to the keypoint stream so teleop "
                "stays usable; set ONERO_START_FALLBACK=0 to require the gate.",
                self.operator_name,
                self._start_fallback_s,
            )
        return robots.ARM_TELEOP_CONT

    def _hand_stream_alive(self) -> bool:
        """Whether the App is pushing tracked hand frames right now."""
        if self._raw_keypoints_subscriber is None:
            return False
        frame = self._raw_keypoints_subscriber.recv_keypoints()
        keypoints = getattr(frame, "keypoints", None)
        if keypoints is None:
            return False
        try:
            joints = np.asarray(keypoints, dtype=float).reshape(-1, 3)
        except (TypeError, ValueError):
            return False
        if joints.size == 0:
            return False
        # Untracked joints stream as the origin; any tracked joint proves the App
        # is pushing and the headset still sees the hand.
        return bool(np.any(joints))

    # -----------------------------------------------------------------------------
    # Gripper
    # -----------------------------------------------------------------------------
    # The App tests only the index, middle and ring tips (GestureDetectorXR
    # .StreamPauser), so the little finger is the one free channel: pinching
    # thumb + little finger can never flip relative/absolute nor latch the pause
    # state, which is exactly what a gripper trigger must not do.
    #
    # Thresholds are wider than the App's 0.02 m index threshold because the little
    # finger is short and the reachable pinch gap is smaller.
    #
    # Tuned on the rig. At some hand angles -- palm rotated away from the headset,
    # little finger partly occluded -- tracking reports the two tips further apart
    # than they physically are, and a 0.03 m close threshold simply never fired:
    # the tips were touching while the estimated span read ~0.04 m. 0.045 m closes
    # reliably at every angle tested. The open threshold keeps the same 0.02 m
    # hysteresis band, and a resting hand measures ~0.08-0.10 m, so the wider band
    # still does not close on its own.
    # Re-tune on the rig: too tight -> never closes, too loose -> closes at rest.
    _PINCH_CLOSE_M = 0.045
    _PINCH_OPEN_M = 0.065
    _THUMB_TIP_IDX = 5
    _PINKY_TIP_IDX = 25
    # Debounce. Headset hand tracking jitters while the hand moves: the
    # thumb/pinky span briefly reads wide for a single frame, which used to open
    # the gripper in the middle of a pinch and drop whatever it held. Median over
    # three samples kills a lone spike, and a state change additionally needs the
    # condition to hold for consecutive frames. Opening needs more evidence than
    # closing because an accidental open is the disruptive direction.
    _PINCH_MEDIAN_WINDOW = 3
    _PINCH_OPEN_FRAMES = 3
    _PINCH_CLOSE_FRAMES = 2

    def _apply_retargeted_angles(self):
        """Run the inherited retargeting, then refresh the gripper state.

        While the start gate is closed the inherited pipeline is skipped entirely.
        That is about more than publishing: its reset branch blocks until a hand
        frame arrives and rebaselines the hand-to-arm mapping, so running it on the
        menu screen printed ``RESETTING TELEOP`` at launch, captured a baseline with
        the hand wherever it happened to be, and then captured a second one after
        Start. Holding ``is_first_frame`` instead means the one and only reset
        happens on the first CONT frame, with the hand placed as the operator
        intends.
        """
        # Diagnostic counters (ONERO_TELEOP_DEBUG=1): _entered counts loop turns,
        # _completed counts turns where the inherited pipeline actually returned.
        # A growing gap between them means the shared pipeline is blocking, which
        # is exactly how a missing subscription looks from the outside: no logs,
        # no commands, and a live process.
        self._entered += 1
        self._cycle_state = None
        try:
            if self._require_start and self._get_arm_teleop_state() == robots.ARM_TELEOP_STOP:
                # Gate closed: no reset, no command. The gripper keeps working --
                # it is a separate effector and behaves the same way when the App
                # pauses -- so its state stays in sync with the hardware.
                self.arm_teleop_state = robots.ARM_TELEOP_STOP
                self.is_first_frame = True
                self._completed += 1
                self._debug_report()
                self._update_gripper()
                return

            super()._apply_retargeted_angles()
            self._completed += 1
            # Report before the gripper step so a future stall in the gripper path
            # shows up as completed growing while the log keeps ticking.
            self._debug_report()
            self._update_gripper()
        finally:
            self._cycle_state = None

    def _pinch_distance(self, keypoints: np.ndarray) -> Optional[float]:
        """Thumb-to-pinky tip span in metres, or ``None`` when a tip is missing.

        Both tips must be present. A joint the headset failed to track streams as
        the origin, so measuring the span against it yields the distance to the
        wrist-frame origin -- 5 to 10 cm, i.e. "fingers far apart". That is what
        used to open the gripper mid-pinch whenever tracking dropped a fingertip
        while the hand was moving.
        """
        if keypoints.shape[0] <= self._PINKY_TIP_IDX:
            return None
        thumb = keypoints[self._THUMB_TIP_IDX]
        pinky = keypoints[self._PINKY_TIP_IDX]
        if not np.isfinite(thumb).all() or not np.isfinite(pinky).all():
            return None
        if not thumb.any() or not pinky.any():
            return None
        return float(np.linalg.norm(thumb - pinky))

    def _classify_pinch(self, distance: float) -> Optional[bool]:
        """Decide the gripper state from one distance sample.

        Returns the new state, or ``None`` when it should not change. The
        hysteresis band (``_PINCH_CLOSE_M`` .. ``_PINCH_OPEN_M``) stops a hand
        resting between the thresholds from oscillating; the median and the
        consecutive-frame requirement reject the single-frame spikes that hand
        tracking produces while the hand moves.
        """
        self._pinch_history.append(distance)
        if len(self._pinch_history) > self._PINCH_MEDIAN_WINDOW:
            self._pinch_history.pop(0)
        # Median over whatever has arrived: waiting for a full window would add
        # two frames of latency to the first pinch after start-up.
        filtered = float(np.median(self._pinch_history))

        if filtered < self._PINCH_CLOSE_M:
            self._pinch_close_streak += 1
            self._pinch_open_streak = 0
            if self._pinch_close_streak < self._PINCH_CLOSE_FRAMES:
                return None
            wanted_closed = True
        elif filtered > self._PINCH_OPEN_M:
            self._pinch_open_streak += 1
            self._pinch_close_streak = 0
            if self._pinch_open_streak < self._PINCH_OPEN_FRAMES:
                return None
            wanted_closed = False
        else:
            self._pinch_close_streak = 0
            self._pinch_open_streak = 0
            return None

        if wanted_closed == self._gripper_is_closed:
            return None
        return wanted_closed

    def _update_gripper(self):
        """Map this hand's thumb/pinky distance to a two-state gripper command.

        Each operator reads its own hand: the right operator drives the right
        gripper from the right hand, the left operator the left one. The pinky
        is used instead of the index finger because thumb+index/middle/ring are
        already the App's mode-switch gestures.
        """
        if self._raw_keypoints_subscriber is None:
            return

        # ``recv_keypoints`` returns the background thread's cached *latest*
        # frame and never clears it, so it always yields the same non-None value.
        # Looping on it until None therefore never terminates -- read it once.
        frame = self._raw_keypoints_subscriber.recv_keypoints()
        if frame is None or frame.keypoints is None:
            return

        self._kp_frames += 1

        keypoints = np.asarray(frame.keypoints, dtype=float).reshape(-1, 3)
        distance = self._pinch_distance(keypoints)
        if distance is None:
            # Hand, or one of the two tips, was not tracked. Hold the last state
            # and keep it out of the history: a dropped joint reads as zero and
            # would otherwise poison the median.
            return

        wanted_closed = self._classify_pinch(distance)
        if wanted_closed is None:
            return

        self._gripper_is_closed = wanted_closed
        # ``publish`` may raise (publisher thread failed to bind the port). The
        # operator loop turns any exception into a ``break`` (operator_base.stream),
        # which would silently end teleoperation on the first pinch -- the arm then
        # stops moving for good. Keep the gripper best-effort instead.
        try:
            self._gripper_publisher.publish(
                host=self._host,
                port=self._gripper_command_port,
                topic="gripper",
                # ``timestamp_s`` is a required field: omitting it raised
                # TypeError on every pinch, so nothing ever reached the arm.
                data=SessionCommand(
                    timestamp_s=time.time(),
                    command="close" if wanted_closed else "open",
                ),
            )
        except Exception as exc:
            logger.error(
                f"{self.operator_name}: failed to publish gripper "
                f"{'close' if wanted_closed else 'open'}: "
                f"{type(exc).__name__}: {exc}"
            )

    def _debug_report(self):
        """Periodic state dump, enabled with ONERO_TELEOP_DEBUG=1."""
        if not self._debug:
            return
        now = time.time()
        if now - self._debug_last < 2.0:
            return
        self._debug_last = now
        logger.warning(
            "[DBG %s] entered=%d completed=%d state=%s is_first_frame=%s "
            "robot_init=%s hand_init=%s kp_frames=%d"
            % (
                self.operator_name,
                self._entered,
                self._completed,
                self.arm_teleop_state,
                self.is_first_frame,
                self.robot_init_h is not None,
                self.hand_init_h is not None,
                self._kp_frames,
            )
        )

    # -----------------------------------------------------------------------------
    # Calibration
    # -----------------------------------------------------------------------------
    @staticmethod
    def _calibration_from_baselines(
        robot_init_h: np.ndarray, hand_init_h: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve the calibration pair for the supplied reset baselines.

        Args:
            robot_init_h: Flange pose measured at reset, in the frame the robot
                publishes on ``endeff_homo`` (world, for the A1 adapter).
            hand_init_h: Hand frame captured at the same instant.

        Returns:
            ``(h_r_v, h_t_v)``, both 4x4 with a purely rotational upper-left 3x3,
            such that the inherited pipeline maps hand motion onto the fixed
            Unity->A1 axis map and leaves the commanded orientation on the
            measured baseline at reset.
        """
        r_init = np.asarray(robot_init_h, dtype=float)[:3, :3]
        r_hand = np.asarray(hand_init_h, dtype=float)[:3, :3]

        # One matrix covers both commands. The pipeline left-multiplies by
        # R_robot_init and post-multiplies the conjugated rotation onto it, so the
        # solved form is R_robot_init^T @ W @ R_hand -- undo the baseline here and
        # the baseline puts the world-frame result back on the outside:
        #   p_cmd = p_init + R_init @ M @ R_hand^T @ d = p_init + W @ d
        #   R_cmd = R_init @ M @ R_rel @ M^-1          = W R_world W^T @ R_init
        # The second line is a world-frame rotation applied to the measured
        # orientation (hand yaw -> tool yaws about the vertical), and at reset
        # (R_rel = I) it collapses to R_init, so the first command never asks the
        # arm to rotate anywhere.
        m = r_init.T @ _UNITY_TO_ROBOT @ r_hand

        h = np.eye(4)
        h[:3, :3] = m.T  # orthogonal matrix: transpose == inverse
        return h.copy(), h.copy()

    def _reset_teleop(self) -> Optional[np.ndarray]:
        """Re-baseline teleop and re-derive the calibration for this session.

        Both baselines are only known here, and both change the direction map, so
        the calibration is solved at the same instant rather than configured.
        Everything else -- requesting the robot pose, capturing the hand frame,
        resetting the filter -- is inherited unchanged.
        """
        first_hand_frame = super()._reset_teleop()
        if first_hand_frame is None or self.robot_init_h is None or self.hand_init_h is None:
            return first_hand_frame

        self.h_r_v, self.h_t_v = self._calibration_from_baselines(
            self.robot_init_h, self.hand_init_h
        )
        logger.info(
            "[%s] calibration re-derived from the measured baselines "
            "(flange rotation kept: first command stays on it)",
            self.operator_name,
        )
        return first_hand_frame
