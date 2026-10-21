"""Read-only joint-limit check for Onero A1 arms.

Teleoperation refuses to start when a measured joint sits outside the URDF
limits by more than the servo-settling tolerance, and the error names only the
largest violation:

    ValueError: IK feedback exceeds URDF limits by 0.165727 rad (tolerance 0.02)

This script prints every joint next to its limits so the offending axis is
visible before starting teleop. It only reads feedback: it does not enable the
motors, send commands, or change the arm state.

Usage:
    python -m beavr.scripts.check_onero_joint_limits --laterality=bimanual
    python -m beavr.scripts.check_onero_joint_limits --laterality=right
"""

import argparse
import sys

import numpy as np

from beavr.teleop.common.configs.loader import Laterality
from beavr.teleop.configs.constants import robots

# Same tolerance the SDK's IK reset applies before it refuses feedback.
OVERSHOOT_TOLERANCE = 0.02


def _side_config(laterality: Laterality):
    """Return (right, left) entries; an entry is None when not requested."""
    right = left = None
    if laterality in (Laterality.RIGHT, Laterality.BIMANUAL):
        right = (
            "right",
            robots.ONERO_DEFAULT_MODEL,
            robots.ONERO_DEFAULT_CAN_CHANNEL,
            robots.ONERO_DEFAULT_MOUNTING,
        )
    if laterality in (Laterality.LEFT, Laterality.BIMANUAL):
        left = (
            "left",
            robots.ONERO_LEFT_DEFAULT_MODEL,
            robots.ONERO_LEFT_CAN_CHANNEL,
            robots.ONERO_LEFT_DEFAULT_MOUNTING,
        )
    return right, left


def _check_arm(side, model, channel, mounting) -> bool:
    """Print the per-joint table for one arm. Returns True when it is within limits."""
    from onero_sdk import Arm
    from onero_sdk.ik_solver import PinkIKSolver

    print(f"\n=== {side}: {model} @ {channel} (mounting={mounting}) ===")
    arm = Arm(model, channel=channel, mounting=mounting)
    try:
        q = np.array(arm.get_state().joints)
        # ``_find_urdf`` is the only route to the URDF the SDK itself loads for
        # IK; there is no public accessor, so the limit check mirrors it here.
        ik = PinkIKSolver(arm._find_urdf())
        lower = np.asarray(ik._model.lowerPositionLimit)
        upper = np.asarray(ik._model.upperPositionLimit)
    finally:
        arm.close()

    # A motor that never answers leaves the backend's cached feedback at its
    # zero default, which would read as "all joints centred" and pass the check.
    # Refuse to report success on data that was never received.
    if not np.any(q):
        print(
            "no joint feedback received (all joints read 0.000): the arm is "
            "probably unpowered or disabled.\n"
            "Power the arm and re-run; this result cannot be trusted."
        )
        return False

    violation = np.maximum(lower - q, q - upper)
    print(f"{'joint':>5} {'measured':>10} {'deg':>8} {'lower':>9} {'upper':>9} {'overshoot':>11}")
    for i in range(len(q)):
        flag = "  <-- OUT" if violation[i] > 0 else ""
        print(
            f"{i:>5} {q[i]:>10.4f} {np.rad2deg(q[i]):>8.2f} {lower[i]:>9.4f} "
            f"{upper[i]:>9.4f} {np.rad2deg(violation[i]):>10.2f}deg{flag}"
        )

    try:
        ik.reset(q, overshoot_tolerance=OVERSHOOT_TOLERANCE)
    except ValueError as exc:
        print(f"reset(): FAILED -- teleop would refuse this arm\n  {exc}")
        return False

    print("reset(): OK -- within limits, teleop can start")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--laterality",
        default="bimanual",
        choices=[item.value for item in Laterality],
        help="which arm(s) to read (default: bimanual)",
    )
    args = parser.parse_args()

    right, left = _side_config(Laterality(args.laterality))
    ok = True
    for entry in (right, left):
        if entry is None:
            continue
        try:
            ok &= _check_arm(*entry)
        except Exception as exc:  # noqa: BLE001 - report, do not traceback
            print(f"{entry[0]}: {type(exc).__name__}: {exc}", file=sys.stderr)
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
