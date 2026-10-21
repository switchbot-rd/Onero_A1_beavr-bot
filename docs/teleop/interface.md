# Interfaces

Robot specific adapters live under `beavr.teleop.interfaces`. Each adapter implements the `RobotWrapper` API providing methods to query state and send commands.

Available interfaces include:

- `LeapHandRobot` – communicates with a simulated or real Leap hand
- `XArm7Robot` – interface for xArm arms
- `RX1RightRobot` – interface for the RX-1 humanoid arm
- `OneroRobot` – interface for Onero A1 arms over SocketCAN

Interfaces are selected through the `--robot_name` argument. Multiple robots can be combined by comma separating their names.

### Onero A1

Onero arms are addressed by their SocketCAN interface name instead of an IP address,
and are driven through the [`onero-sdk`](https://pypi.org/project/onero-sdk/) package:

```bash
python teleop.py --robot_name=onero --laterality=right
```

#### Prerequisites

Install the SDK with the inverse-kinematics extra, required by real-time Cartesian
servo:

```bash
pip install "onero-sdk[ik]"
```

Bring the CAN interface up *before* teleoperation starts. The A1 adapters are
USB-serial (SLCAN) devices and need a `slcand` bridge first:

```bash
sudo slcand -o -s8 -t hw -S 1000000 /dev/ttyACM0 can0 && sudo ip link set up can0
```

Interface naming depends on the SDK's udev rules — with them installed the arms
appear as `slcan0` (right) and `slcan1` (left).

**Three values describe the installation rather than this repository.** Set all
three, or the arm will not start, or will not track correctly:

| Value | Read by | How to set it |
|---|---|---|
| CAN interface name | `OneroRobotCfg.can_channel` | `ONERO_CAN_CHANNEL=slcan0`, and `ONERO_LEFT_CAN_CHANNEL` for the left arm |
| Network host | every component's `host` | `BEAVR_HOST_ADDRESS=10.8.80.204` (read in `configs/constants/network.py`) |
| Loop rate | every component's timer | `VR_FREQ` in `configs/constants/robots.py` |

The host address must be an address that exists on the machine running teleop,
because publishers bind `tcp://*:<port>` while every subscriber connects to that
address, and the headset app talks to the same address. When it names an
unreachable address the failure is silent (see Troubleshooting), so set it
explicitly rather than relying on the default captured from the validated cell.

`--teleop.network.host_address` and `--teleop.control.vr_freq` are parsed,
validated and logged, and then ignored: the components read the constants
directly rather than their own config. Environment variables are the working
route.

The remaining notes are about correctness rather than startup:

- **Set `with_gripper=True` on gripper-equipped arms.** The SDK's gripper gravity
  profile scales the modelled per-joint gravity torque by up to 5.3x (J5) and
  2.8x (J3); leaving it `False` under-compensates gravity and the joints sag
  noticeably under load. Declare a measured payload with `Arm.set_payload()`
  when the gripper mass and centre of gravity are known.
- **Set `mounting` explicitly; never leave it `None`.** `resolve_mounting(None)`
  silently returns `UPRIGHT`, which places the modelled gravity along base `-Z`.
  The A1 is mounted sideways (see the module docstring of the SDK's
  `mounting.py`), so on a real cell `UPRIGHT` rotates the gravity vector by 90
  degrees: the feedforward pushes sideways instead of up, the joints lag under
  load, and a servo hold against a rigid support becomes a sustained stall --
  audible screech and motor heating instead of tracking. The validated A1 cell
  in this repo uses `humanoid_right` (A1R) and `humanoid_left` (A1L); those
  presets put base `+X` up and base `+Z` horizontally outward, so the first
  joint swings the whole arm in the vertical plane.
- **Leave `kp`/`kd` unset, so the SDK's own `50` / `1.0` applies.** The uniform
  `kp=50` lets the joints lag 0.7-1.6 deg behind the commanded angle — 5-25 mm at
  the tool through the ~5.6 mm/deg lever arm at J4 — and measured here that showed
  up as 11.8 mm of steady-state tracking error with only 2 of the 7 joints moving
  at all. Raising it to `kp=150` cut the error to 4.8 mm and brought 5 of 7 joints
  in, but left J1-J4 under-damped, since their calibrated `kd` is 4.0 and not 1.0,
  and the arm jittered in continuous teleoperation. Setting `kd=4` to match instead
  over-damps J5-J7 four-fold, whose calibrated value is 1.0, and screeches. **No
  flat pair is correct, because `servoj` broadcasts one scalar while the model is
  per-joint.** The calibrated values `(150, 150, 150, 150, 40, 40, 40)` /
  `(4, 4, 4, 4, 1, 1, 1)` already exist in the SDK registry but are unreachable
  from its servo entry points; per-joint support there is the actual fix.
- **The SDK's 0.02 rad joint-limit tolerance is widened to 0.1 rad** — see
  `OneroRobot._relax_ik_tolerances` and `ONERO_IK_LIMIT_OVERSHOOT_RAD`.
  `servop` re-seeds its IK from the *measured* joints every tick, and
  `IKSolver.reset` refuses feedback that sits more than 0.02 rad outside the URDF
  range. Servo sag at the edge of the workspace exceeds that: measured 0.0297 rad
  (left) and 0.0335 rad (right), after which `servop` refused every target and
  both arms stayed frozen for ~60 s (1741 consecutive frames) until restart — the
  arm cannot move itself back inside the limit while every command is refused.
  The solver still projects its warm start back inside the limits, so widening
  only removes the deadlock; it does not let the arm drive further.
  Do **not** widen the *convergence* tolerances instead: an unreachable target
  then counts as solved, that pose becomes the next step-clamp baseline, and the
  arm parks at the workspace edge and stops responding (measured: one "holding at
  the limit" line per second for the rest of the session).
- Model, mounting and world-frame options live in `configs/robots/onero_config.py`
  (`OneroRobotCfg`) and in `configs/constants/robots.py`.
- **The VR→robot calibration is derived at teleop reset, not configured.**
  `OneroOperator._reset_teleop()` solves, from the two baselines the reset
  captures (flange pose and hand frame), the matrix that makes the command map
  equal the fixed Unity→A1 axis map (x right / y up / z forward → X forward /
  Y left / Z up). Both `H_R_V` and `H_T_V` receive that one matrix; the module
  constants in `onero_operator.py` are only the pre-reset values. Consequences
  worth knowing before touching them:
  - The map no longer depends on where the arm happens to start, nor on how the
    hand is held at reset. Hard-coding a matrix instead (the xArm7 approach)
    makes the direction map rotate with the flange orientation at reset, which
    fails silently: with the xArm7 placeholders on an A1 cell, moving the hand up
    commanded the arm backwards.
  - The commanded orientation at reset equals the measured flange orientation,
    so the first command never asks the arm to slew. Blanking the baseline
    rotation -- the obvious way to make the map pose independent -- does ask for
    exactly that, by 0.96 in the measured case, and is wrong for that reason.
  - Rotation is composed in the world frame: a hand yaw turns the tool about the
    vertical.

#### Safety notes

**A `home` command does not send the arm to zero.** `Arm.home()` drives all seven
joints to zero, and on the cell this adapter was developed against that pose puts
the lowest link 247 mm below the plane the arm rests on — the straight-line path
there crosses that plane after roughly 5% of the motion. The control loop calls
`home()` whenever the operator requests it, so routing it to zero made "home" a
motion into the table. `home()` therefore travels to `OneroRobotCfg.park_pose_deg`
instead, which defaults to the pose derived for that cell and is overridable
(`ONERO_PARK_POSE_DEG="deg,deg,..."`). **Check that the pose suits your cell
before relying on it**, and re-derive it if the arm is mounted differently. A
requested travel of more than 170 deg on any joint is refused rather than acted
on, since that is what a slipped decimal point looks like.

**Starting teleoperation does not move the arm.** `home_on_start` is off by
default, so `stream()` begins tracking from the measured pose; nothing else in
this ecosystem homes on startup either. Enable it only on a cell where you want
the arm to travel before tracking.

**A component that crashes leaves the arm holding torque, and that is
deliberate.** The A1 has no brake that can hold it: `Arm.disable()` drops it under
gravity, and measured four times on this cell the lowest link fell 5.3-6.6 deg,
about 41 mm, coming to rest 6-7 mm above the table from a parked pose 47-54 mm up.
From a low pose — a demo at a few centimetres — that same drop lands on the table.
Holding torque is the safer side of the trade, and it is the side the SDK already
chose: `Arm.close()` defaults to `disable=False`, and `enable()` skips its
re-enable sequence "to preserve holding torque". Franka can power down after a
stop because it has brakes; this arm cannot. **A crashed session therefore leaves
an energised, unsupervised arm** — watch the log for the component's exit, and
recover by restarting teleoperation rather than by powering the arm down. If that
ever needs changing, the fix is a controlled stop (IEC 60204-1 category 1) — the
parent usually survives, and once the robot interface dies the CAN channel is
free, so it can reconnect, lower the arm to a resting pose, and only then disable
— not a bare `disable()` on the way out.

#### Troubleshooting

**The operator never leaves its reset handshake, and the robot never answers.**
The host address does not reach a machine that is listening.
`HOST_ADDRESS` in `configs/constants/network.py` still names the cell the code
was written against. This failure leaves no trace, which is what makes it worth
documenting: PUB sockets bind `0.0.0.0` and look healthy, the process stays up,
and no connection error is raised anywhere — every SUB simply connects to an
address that answers nothing, so the operator publishes `reset` forever and never
reaches `endeff_coords`. Check the host first: compare `HOST_ADDRESS` against the
address of the machine running the robot interface. Note that `ss -ltnp` will not
reveal this — every component binds `0.0.0.0` and looks healthy.

**`Cannot connect to CAN channel 'can0': Interface not available`.**
The interface name does not match the host. A rig wired through a serial bridge
enumerates as `slcan0`, not `can0`. Set `ONERO_CAN_CHANNEL` to the name that
`ip link` reports. The same error appears if the interface exists but is down —
run the `slcand` command from Prerequisites.

**`[Errno 105] No buffer space available` after roughly a minute of teleoperation.**
The CAN transmit queue is full. At 100 Hz each `servoj` writes 7 motor frames,
so the link carries ~700 frames/s, and an `slcan` interface defaults to
`txqueuelen=10`. Raise the queue, or lower the rate:

```bash
sudo ip link set slcan0 txqueuelen 1000
```

A robot interface that dies this way exits with a non-zero code and takes the CAN
channel with it, so the session has to be restarted.

**The arm follows, but only the wrist and elbow move, and it lags by ~10 mm.**
This is the friction floor at the SDK's uniform `kp=50`, not a misconfiguration:
each joint needs `friction_torque / kp` of commanded offset before it moves at
all, which on J1 is about 1.7 deg — wider than the share the inverse kinematics
gives it for a modest Cartesian move. The joints that do have enough leverage
move; the rest do not. See the `kp`/`kd` note under Prerequisites for why no flat
gain fixes it.
