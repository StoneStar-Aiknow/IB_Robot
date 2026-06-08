#!/bin/bash
# scripts/start_rosclaw.sh
# Script to build and start the RosClaw social bridge independently.

set -eo pipefail

# Load project environment
if [ -f ".shrc_local" ]; then
    source .shrc_local
else
    echo "⚠️  .shrc_local not found. Please run this from the workspace root."
    exit 1
fi

echo "RosClaw: Building rosclaw packages..."
colcon build --merge-install --packages-select rosclaw_discovery rosclaw_msgs rosclaw_agent --base-paths src/rosclaw/ros2_ws/src

echo "RosClaw: Sourcing workspace..."
if [[ ! -f "install/setup.bash" ]]; then
    echo "ERROR: install/setup.bash not found. Build may have failed."
    exit 1
fi
source install/setup.bash

echo "RosClaw: Starting Rosbridge WebSocket server on port 9090..."
# Run rosbridge in the background
ros2 run rosbridge_server rosbridge_websocket &
ROSBRIDGE_PID=$!

echo "RosClaw: Starting RosAPI node..."
# Run rosapi to provide /rosapi/* services required by frontend
ros2 run rosapi rosapi_node &
ROSAPI_PID=$!

# Give them a moment to start
sleep 2

echo "RosClaw: Starting RosClaw Discovery node..."
# Run discovery node in the foreground
ros2 run rosclaw_discovery discovery_node

# Cleanup background process on exit
trap "echo 'Shutting down RosClaw...'; kill $ROSBRIDGE_PID $ROSAPI_PID" EXIT
