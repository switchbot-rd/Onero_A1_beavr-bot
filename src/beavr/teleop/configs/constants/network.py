# -----------------------------------------------------------------------------
# Network Configuration Constants
# -----------------------------------------------------------------------------

import os

# Host addresses
#
# Every component subscribes by connecting to this address while publishers bind
# to `tcp://*:<port>`, so it must be an address that exists *on this machine*.
# Pointing it at another host does not degrade gracefully: the subscribers connect
# into the void, the operator's teleop reset never receives endeff_homo and blocks
# forever, and the detector dies at startup with
# `ZMQError: Cannot assign requested address`.
#
# The default is the Wi-Fi address of the validated A1 cell, not a portable
# default. On any other machine override it with this host's own LAN address
# before starting teleop, e.g.
#     export BEAVR_HOST_ADDRESS=$(hostname -I | awk '{print $1}')
# and point the headset app at the same address.
HOST_ADDRESS = os.environ.get("BEAVR_HOST_ADDRESS", "10.8.80.204")

# Robot IP addresses
LEFT_XARM_IP = "192.168.1.237"
RIGHT_XARM_IP = "192.168.1.197"
LEFT_ARM_IP = "192.168.86.216"  # For Left XArm
RIGHT_ARM_IP = "192.168.86.230"  # For Right XArm

# LEAP hand USB ports
LEFT_LEAP_USB_PORT = "/dev/ttyUSB1"
RIGHT_LEAP_USB_PORT = "/dev/ttyUSB0"

# Robot IP addresses (alternative naming)
XARM_LEFT_IP = "192.168.1.237"
XARM_RIGHT_IP = "192.168.1.197"

# Network configuration
TELEOP_HANDSHAKE_PORT = 8150

# Deployment constants
DEPLOY_REACH_THRESHOLD = 0.35
DEPLOY_FREQ = 3

# Teleop constants
ARM_TELEOP_CONT = 1
ARM_TELEOP_STOP = 0

# Resolution specific parameters
ARM_HIGH_RESOLUTION = 1  # for arm teleoperation
ARM_LOW_RESOLUTION = 0

LEFT_HAND_PORT = 8110
RIGHT_HAND_PORT = 8087
