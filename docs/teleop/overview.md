# Overview of teleop system

## Teleoperation Architecture Overview

This document explains how the BeavR teleoperation system hangs together end‑to‑end for a first‑time reader. It uses a concrete run example and walks through configuration, networking, class roles, and the data flow from the VR detector to the robot controller.

Example run:

```bash
python teleop.py --robot_name=xarm7,leap --laterality=right
```

- This starts the teleop entrypoint, selecting two robots: `xarm7` (an arm) and `leap` (a hand) with the right side active. The system composes both robots’ components (detectors, transforms, operators, robots) and launches them as processes/threads.

### TL;DR data flow

1) Oculus VR detector reads hand data and publishes keypoints over ZeroMQ.
2) Transform component converts raw keypoints to a stable 6DoF hand frame in the VR/world coordinates and republishes it.
3) Operator (e.g., `XArm7RightOperator`) consumes the transformed frame, computes a target end‑effector pose using calibrated transforms, optionally filters it, and publishes commands.
4) Robot interface (`XArm7Robot`) subscribes to those commands and calls the low‑level controller (`DexArmControl`) to move real hardware. It also publishes robot state for observers/recorders.
5) A lightweight handshake/ACK mechanism coordinates critical state transitions (e.g., pause/reset) between publishers and subscribers.

---

## 1) Entry point: `src/beavr/teleop/main.py`

What it does:
- Defines a `MainConfig` dataclass that nests the structured `TeleopConfig` (global teleop settings) and robot selection arguments.
- Uses `draccus` to auto‑generate CLI flags from dataclasses. Precedence is: CLI flags > YAML file overrides > defaults.
- Loads a YAML config (default `config/dev.yaml`) and merges it underneath any CLI flags.
- Builds the requested robot configurations by name and laterality, then instantiates the teleoperation system (`TeleOperator`) and starts all its processes.

Key fields in `MainConfig`:
- `teleop: TeleopConfig`: structured teleop config (control loop rates, network ports, flags, etc.).
- `robot_name: str`: comma‑separated robot names (e.g., `xarm7,leap`).
- `laterality: str`: `right | left | bimanual`.
- `config_file: str`: path to YAML overrides.
- `robot: Any`: the composite built robot structure (set in `__post_init__`).

Execution flow:
- `main()` is wrapped by `@draccus.wrap()` so CLI flags map to dataclass fields.
- `load_yaml_config()` reads the YAML; `apply_yaml_preserving_cli()` merges it, preserving any CLI overrides.
- `load_robot_config(robot_name, laterality)` returns a composite configuration for the requested robots.
- `TeleOperator(config)` creates components, then `get_processes()` returns the processes to start.

What your example command means:
- `--robot_name=xarm7,leap`: include both robots.
- `--laterality=right`: initialize only right‑hand/right‑arm components (you can use `bimanual` to include both sides).

---

## 2) Robot configs: `leap_config.py`, `xarm7_config.py` and `onero_config.py`

Both files follow the same pattern:
- Small dataclasses that describe component configuration for a robot:
  - `...OperatorCfg`: config for the operator process that turns hand frames into commands.
  - `...RobotCfg`: config for the robot interface process that receives commands and talks to hardware/sim.
  - `...Config`: the top‑level dataclass registered in a config registry; it assembles per‑laterality components: detector(s), transform(s), visualizer(s), operator(s), robot(s), and optional recorder settings.
- A `build()` method constructs concrete component instances from the dataclasses.
- All network constants (host, port numbers, topic names) come from `beavr.teleop.configs.constants`.

### 2.1) `src/beavr/teleop/configs/robots/leap_config.py`

Purpose:
- Configure a Leap (VR hand) robot and its operator for right, left, or bimanual setups.

Important types:
- `LeapHandOperatorCfg`:
  - Ports: subscribes to transformed keypoints; publishes joint/cartesian commands for the hand; publishes reset events.
  - `hand_side`: `right` or `left` chooses the ZMQ topic to subscribe to.
  - `finger_configs` and `logging_config`: feature toggles and optional logging.
  - `build()` returns a `LeapHandOperator` instance.
- `LeapHandRobotCfg`:
  - Ports: subscribes to commands, publishes joint states and robot state for recording, subscribes to reset.
  - `simulation_mode`: switch for sim vs. real.
  - `hand_side`: used to set `is_right_arm` when building the concrete `LeapHandRobot`.
  - `state_publish_port`: separate per side so right and left can run together.
  - `build()` returns a `LeapHandRobot` instance.
- `LeapHandConfig`:
  - Top‑level container that, in `__post_init__`, configures detectors, transforms, visualizers, operators, and robots based on laterality.
  - For `bimanual`, it appends both right and left instances; for a single side, it appends just that side’s components.
  - `build()` creates concrete instances for each configured component.

### 2.2) `src/beavr/teleop/configs/robots/xarm7_config.py`

Purpose:
- Configure an XArm7 robot (left, right, or bimanual) and its operator.

Important types:
- `XArm7RobotCfg`:
  - Network: robot IP per side (`RIGHT_XARM_IP`, `LEFT_XARM_IP`) and per‑side port mapping.
  - Publishes a rich state dictionary for recording (`state_publish_port`).
  - Validates port ranges and IP format in `__post_init__`.
  - `build()` returns an `XArm7Robot` (see interface below).
- `XArm7OperatorCfg`:
  - Subscribes to transformed keypoints and auxiliary topics (button for resolution, pause/teleop state).
  - Publishes end‑effector commands.
  - Picks the concrete class by side at build time:
    - Right: `XArm7RightOperator`
    - Left: `XArm7LeftOperator`
  - Includes optional pose logging configuration.
- `XArm7Config`:
  - Same assembly pattern as Leap: laterality decides which side(s) are appended to detectors, transforms, visualizers, operators, and robots.
  - Uses slightly offset port numbers for the left arm so both arms can run together cleanly.

---

## 3) Messaging and networking: `src/beavr/teleop/utils/network.py`

ZeroMQ is used throughout for real‑time, decoupled pub/sub messaging between processes.

Core pieces:
- Global ZMQ context
  - `get_global_context()` and `set_global_context()` provide a shared context per process.

- Publisher/Subscriber primitives
  - `BasePublisher`: binds a PUB socket (`tcp://*:<port>`); `pub.send_multipart([topic, payload])`.
  - `BaseSubscriber`: a thread that owns a SUB (or other) socket; it `connect()`s to `tcp://host:port`, subscribes to a topic, polls with timeouts, and calls `process_message(data)` when messages arrive.
  - `ZMQKeypointPublisher` / `ZMQKeypointSubscriber`: convenience classes that send/receive pickled keypoint arrays.
  - `ZMQCompressedImageTransmitter` / `ZMQCompressedImageReceiver`: compress and stream camera images.

- Socket helpers
  - `create_push_socket`, `create_pull_socket`, `create_request_socket`, `create_response_socket` wrap common ZMQ patterns with timeouts and error handling for robustness (HWM, linger, recv timeouts, conflation).

- Central publisher manager
  - `ZMQPublisherManager`: a singleton that owns a background `PublisherThread` per `(host, port)` with a thread‑safe queue. This ensures ZeroMQ sockets are only used from the thread that created them and avoids “socket used from multiple threads” issues.
  - Callers use `publish(host, port, topic, data)` and the manager handles serialization, HWM, and backpressure. It also monitors publisher thread health.

- Handshake/ACK coordination
  - `HandshakeCoordinator`: registers subscribers (id → host, port) and runs REP servers on the subscriber side. Publishers can request acknowledgments from a set of subscribers before/after a critical action (`request_acknowledgments([...])`).
  - `publish_with_guaranteed_delivery(...)` optionally combines a publish with an acknowledgment round trip.

- Cleanup
  - `cleanup_zmq_resources()` shuts down publisher threads, handshake servers, and terminates the ZMQ context.

Message format:
- All messages are multipart: `[topic: bytes, payload: bytes]` where `payload` is usually `pickle.dumps(data)`.
- Subscribers filter by topic at subscription time and expose the latest payload via small accessors (e.g., `recv_keypoints()`).

---

### 4.1) VR Detector: `src/beavr/teleop/components/detector/oculus.py`

- `OculusVRHandDetector` (single hand) or `BimanualOculusVRHandDetector` (both hands) reads raw hand data via PULL sockets from preconfigured ports (see `network.RIGHT_HAND_PORT`, `network.LEFT_HAND_PORT`, etc.).
- It parses the raw stream into a numeric list (first element indicates relative vs absolute, followed by XYZ triples).
- It publishes these keypoints via `ZMQPublisherManager` on the `KEYPOINT_STREAM_PORT` with topics:
  - Right hand: `right`
  - Left hand: `left`
  - Resolution button: `button` (mapped to high/low resolution enum)
  - Pause: `pause` (mapped to teleop stop/continue enum)

Note: Transform components (created by the robot config) subscribe to `KEYPOINT_STREAM_PORT` and publish a stabilized “transformed hand frame” to `KEYPOINT_TRANSFORM_PORT`.

### 4.2) Operator base and right‑arm operator

- `src/beavr/teleop/components/operators/xarm_base.py` provides `XArmOperator`, the base class that:
  - Subscribes to the transformed keypoint topic (per side). For the right hand, the topic is typically something like `TRANSFORMED_HAND_FRAME` (right/left variants are handled by side‑specific topics/namespacing).
  - Optionally subscribes to resolution (`button`) and pause (`pause`) topics.
  - Subscribes to the robot’s current pose stream (`endeff_homo`) to capture the initial baseline during reset.
  - Maintains calibrated transforms:
    - `H_R_V`: Robot base → VR base
    - `H_T_V`: Hand‑Tracking base → VR base
  - On each cycle (at `VR_FREQ`):
    1) Checks teleop state (pause/continue) and whether a reset is needed.
    2) If resetting, requests robot pose and captures the initial hand frame; otherwise, takes the latest transformed hand frame.
    3) Computes relative motion of the hand since reset, maps it into the robot base frame using `H_R_V` and `H_T_V`.
    4) Builds a target end‑effector pose, applies an optional complementary filter (position LERP + orientation SLERP), and normalizes the quaternion.
    5) Publishes the command dictionary `{position: [x,y,z], orientation: [qx,qy,qz,qw], timestamp}` to the robot’s command port.
  - Provides a small handshake server so other components can confirm it is reachable.

- `src/beavr/teleop/components/operators/xarm7_right.py` specializes `XArmOperator` with right‑arm‑specific `H_R_V` and `H_T_V` matrices and sets `hand_side=right`. The left‑arm class is analogous.

### 4.3) Robot interface and controller: `src/beavr/teleop/interfaces/xarm7_robot.py`

- `XArm7Robot` adapts teleop commands to the hardware via `DexArmControl`:
  - Subscribes to `endeff_coords` on the command port (end‑effector commands from the operator).
  - Subscribes to `reset` and `home` topics (separate ports) to handle baseline capture and homing.
  - Subscribes to `pause` (teleop state), used to gate motion.
  - When a command arrives, it concatenates position and orientation and calls `self._controller.move_arm_cartesian(...)`.
  - Publishes periodic state dictionaries on a dedicated state port (`state_publish_port`) using the robot’s name as the topic (e.g., `xarm7_right`). The state includes joint states, cartesian states, commanded states, etc., with timestamps suitable for recording/analysis.
  - Exposes utility accessors like `get_cartesian_state_from_operator()` for the recorder.
  - Hosts its own handshake server so control flows can verify connectivity.

- `DexArmControl` (controller)
  - A low‑level API wrapper around the XArm hardware. It provides motion primitives (home, move_cartesian, set modes, query states) that `XArm7Robot` calls.

---

## 5) Configuration, laterality, and composition

- Laterality (`right | left | bimanual`) drives which side(s) are instantiated for each robot’s detectors, transforms, visualizers, operators, and robots.
- Each side uses dedicated topics and, where needed, distinct ports so both sides can run concurrently.
- The top‑level `MainConfig` collects your `--robot_name` list, resolves each robot’s registered config class (e.g., `XArm7Config`, `LeapHandConfig`), and appends the side‑appropriate components to the launch plan.
- `build()` methods on the robot configs create concrete objects (subscribers, publishers, threads) ready to be launched by the teleop runtime.

---

## 6) Operational controls: reset, pause, resolution

- Reset: The operator requests a robot pose sample and captures a new hand baseline; subsequent motion is interpreted relative to that baseline.
- Pause/Continue: A small integer enum (`ARM_TELEOP_STOP` / `ARM_TELEOP_CONT`) gates motion in the operator and on the robot side.
- Resolution: Button presses select “high” vs “low” resolution scaling for hand motion → end‑effector translation.
- Handshake: Critical transitions can be protected by an ACK round trip via `HandshakeCoordinator` to avoid races.

---

## 7) Putting it together for your example

Command: `python teleop.py --robot_name=xarm7,leap --laterality=right`

What starts:
- `LeapHandConfig` (right):
  - VR detector (right‑hand input), transforms, visualizer, Leap operator (right), Leap robot (right).
- `XArm7Config` (right):
  - Detector can be shared if configured bimanually; right transform; right operator (`XArm7RightOperator`); right robot (`XArm7Robot` with `RIGHT_XARM_IP`).

Runtime flow:
1) VR detector publishes raw right‑hand keypoints + button + pause.
2) Transform publishes stable right‑hand frame.
3) `XArm7RightOperator` consumes the frame, applies `H_R_V`/`H_T_V`, filters, and publishes `endeff_coords`.
4) `XArm7Robot` consumes commands and moves the real robot via `DexArmControl`, while publishing pose and state.
5) Optional recorders/visualizers subscribe to state streams.

---

## 8) Tips for debugging and tuning

- Ports and topics: If no data flows, verify the port numbers and topics for your side (right vs left). Left ports are often offset to avoid collisions.
- Filtering: If motion feels jittery or laggy, tweak `CompStateFilter` ratios or disable filtering to compare.
- Pause/Reset: Use the pause and reset controls to re‑baseline your session; watch logs for handshake/ACK messages during transitions.
- Cleanup: On shutdown, components call `cleanup_zmq_resources()` to close sockets/threads; if a port is “in use,” another process may still be running.

---

## 9) Glossary

- Keypoints: 3D landmarks from hand tracking (VR headset). Raw → transformed.
- Transformed hand frame: A stable 6DoF pose derived from keypoints (position + orientation).
- End‑effector (EE): The robot tool pose (Cartesian position + quaternion orientation).
- Laterality: Which side(s) are active (right/left/bimanual).
- ZMQ topic: A string label in PUB/SUB to filter messages (e.g., `right`, `pause`, `endeff_coords`).
