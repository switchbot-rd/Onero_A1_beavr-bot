"""Onero A1 specific VR hand detector.

Three properties of the headset stream need handling that the shared detector does
not provide. All of them are kept here so ``oculus.py`` stays untouched:

* the payload is terminated with a trailing colon (``relative:...,...:``). The
  shared parser splits on ``,`` and ``float("0.3887634:")`` raises, so *every*
  frame was dropped and the arm never tracked;
* the app emits one-shot ``DIAGNOSTIC_TEST_<Hand>_<hh:mm:ss.mmm>`` lines before
  the real stream. Their embedded colons make ``split(":")[1]`` return a single
  value, which surfaces as ``cannot reshape array of size 1 into size (26, 3)``
  and takes the whole transform process down;
* the pause channel carries more than the two codes the shared ``stream`` knows
  about, and the shared code treated *every* non-``Low`` payload as "continue".

The filtering happens in ``_receive_data`` rather than ``_process_keypoints``:
the shared ``stream`` publishes whatever the parser returns, so returning None
there would forward an ``InputFrame(keypoints=None)`` and crash the transform --
the very failure being avoided here.
"""

from __future__ import annotations

import logging
import os
import time

from beavr.teleop.components.detector.vr.oculus import OculusVRHandDetector
from beavr.teleop.configs.constants import robots

logger = logging.getLogger(__name__)

_DIAGNOSTIC_PREFIX = "DIAGNOSTIC"

# The App's pause socket (GestureDetectorXR.SendPauseStatusThroughController)
# sends exactly two state payloads: "High" while *continue* is armed -- which is
# what the Start button sets up -- and "Low" otherwise. Two other kinds of
# traffic share the socket and must never be mistaken for a state:
#
# * ``KEEPALIVE``: NetworkKeepAlive pings every five seconds;
# * ``DIAGNOSTIC_TEST_Pause_<time>``: NetMQController.PerformDiagnosticTests runs
#   one right after the sockets are created, i.e. before Start is pressed.
#
# The shared ``stream`` maps ``b"Low"`` to "resume" and *anything else* to
# "pause". OneroOperator reads "pause" as the start code, so without this
# whitelist both a keep-alive and the diagnostic probe would open the start gate
# on an App that is still sitting on the menu. Only these two payloads are let
# through; the resulting codes are then exactly what the operator expects:
#
#   App "High" (Start / continue) -> published "pause"  -> ARM_TELEOP_CONT
#   App "Low"  (paused / menu)    -> published "resume" -> ARM_TELEOP_STOP
_PAUSE_PAYLOADS = {b"High", b"Low"}

# Set ONERO_VR_DEBUG=1 to see one line every couple of seconds describing what
# arrives from the headset and whether it was forwarded.
_DEBUG_FRAMES = os.environ.get("ONERO_VR_DEBUG") == "1"


class OneroVRHandDetector(OculusVRHandDetector):
    """Detector that only forwards well-formed 26-point frames."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._parsed_frame = None
        self._last_frame_log = 0.0
        # Only the hand channels carry keypoints. Pause and button payloads are
        # short control strings ("High" / "Low" / "KEEPALIVE"), so running them
        # through the keypoint parser rejected every one of them -- the pause
        # channel looked dead and the App's Start never reached the operator.
        self._keypoint_sockets = {
            f"{robots.KEYPOINTS}_{hand_side}" for hand_side in self.hand_ports
        }

    def _receive_data(self, socket_name):
        """Receive a frame and drop it unless it is valid for that channel."""
        data = super()._receive_data(socket_name)
        self._parsed_frame = None

        if data is None:
            return None

        if socket_name == robots.PAUSE:
            # Only the two state payloads are forwarded; see _PAUSE_PAYLOADS.
            if (data or b"").strip() not in _PAUSE_PAYLOADS:
                return None
            return data

        if socket_name not in self._keypoint_sockets:
            return data

        values = self._parse_keypoints(data)
        if values is None:
            # Never let a diagnostic or malformed frame reach the publisher.
            return None

        self._parsed_frame = values
        self._log_frame(socket_name, data, len(values))
        return data

    def _process_keypoints(self, data):
        """Return the parse result cached by ``_receive_data`` for this frame."""
        if self._parsed_frame is not None:
            values, self._parsed_frame = self._parsed_frame, None
            return values
        return self._parse_keypoints(data)

    def _log_frame(self, socket_name, data, n_values):
        if not _DEBUG_FRAMES:
            return
        now = time.time()
        if now - self._last_frame_log <= 2.0:
            return
        self._last_frame_log = now
        raw = data.decode(errors="replace").strip()
        logger.warning(
            "[onero-vr] %s len=%d colons=%d points=%d -> %d values",
            socket_name,
            len(data),
            raw.count(":"),
            raw.count("|") + 1,
            n_values,
        )

    def _parse_keypoints(self, data):
        """Parse a keypoint frame into a flat list, or None if it is not one."""
        data_str = data.decode(errors="replace").strip()

        # Connectivity probes from the app are not keypoint frames.
        if not data_str or data_str.startswith(_DIAGNOSTIC_PREFIX):
            return None

        if ":" not in data_str:
            return None

        # Strip the trailing delimiter the stream appends.
        payload = data_str.split(":", 1)[1].strip().strip(":")
        coords = payload.split("|")
        if len(coords) != robots.OCULUS_NUM_KEYPOINTS:
            logger.debug(
                "Dropping %s frame with %d points (expected %d)",
                self.hand_config,
                len(coords),
                robots.OCULUS_NUM_KEYPOINTS,
            )
            return None

        values = []
        for coord in coords:
            parts = coord.split(",")[:3]
            if len(parts) != 3:
                return None
            try:
                values.extend(float(val.strip().strip(":")) for val in parts)
            except ValueError:
                return None

        return values
