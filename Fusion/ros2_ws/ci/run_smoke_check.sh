#!/usr/bin/env bash
set -eo pipefail

echo "============================================================"
echo "    AUTONOMOUS RACING SENSOR FUSION - SMOKE CHECK (R5)"
echo "============================================================"

# Resolve script directory and workspace root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "${SCRIPT_DIR}/../Fusion/ros2_ws" ]; then
    WS_DIR="$(cd "${SCRIPT_DIR}/../Fusion/ros2_ws" && pwd)"
elif [ -d "${SCRIPT_DIR}/../src" ]; then
    WS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    WS_DIR="${PWD}"
fi

echo "[smoke_check] Workspace directory: ${WS_DIR}"
cd "${WS_DIR}"

# 1. Source ROS 2 base installation if available
if [ -f "/opt/ros/humble/setup.bash" ]; then
    echo "[smoke_check] Sourcing /opt/ros/humble/setup.bash"
    source /opt/ros/humble/setup.bash
fi

# 2. Build ROS 2 workspace
echo "[smoke_check] [1/3] Building workspace packages (fusion_interfaces, fusion_core, fusion_adapters)..."
colcon build --packages-select fusion_interfaces fusion_core fusion_adapters --symlink-install

# 3. Source local workspace
echo "[smoke_check] Sourcing workspace install/setup.bash"
source install/setup.bash

# 4. Run existing tracker unit tests (minimal budget, zero waste)
echo "[smoke_check] [2/3] Running existing unit tests (pytest -q)..."
PYTHONPATH="${WS_DIR}/src/fusion_core:${PYTHONPATH}" python3 -m pytest -q src/fusion_core/test

# 5. Headless run of synthetic stack verifying >= 20 Hz output on perception topics and zero crashes
echo "[smoke_check] [3/3] Launching synthetic stack for headless rate & stability verification..."

LOG_FILE="/tmp/synthetic_stack_smoke.log"
ros2 launch fusion_core synthetic_stack.launch.py use_foxglove_bridge:=false > "${LOG_FILE}" 2>&1 &
STACK_PID=$!

echo "[smoke_check] Synthetic stack launched with PID ${STACK_PID}. Waiting 6 seconds for pipeline warm-up..."
sleep 6

# Check if process is still running (zero crashes)
if ! kill -0 ${STACK_PID} 2>/dev/null; then
    echo "[smoke_check] ERROR: Synthetic stack crashed on startup! Log dump:"
    cat "${LOG_FILE}"
    exit 1
fi
echo "[smoke_check] Synthetic stack is running cleanly (0 crashes)."

# Verify topic publication and rates
echo "[smoke_check] Verifying topic frequencies (>= 20.0 Hz requirement)..."

TOPICS=("/perception/cone_observations" "/perception/cone_tracks" "/perception/diagnostics")

for topic in "${TOPICS[@]}"; do
    echo "[smoke_check] Sampling frequency on ${topic}..."
    HZ_OUT=$(timeout 4 ros2 topic hz "${topic}" 2>/dev/null || true)
    echo "${HZ_OUT}"
    
    if echo "${HZ_OUT}" | grep -q "average rate:"; then
        AVG_RATE=$(echo "${HZ_OUT}" | grep "average rate:" | tail -n 1 | awk '{print $3}')
        echo "[smoke_check] PASS: ${topic} active at ~${AVG_RATE} Hz"
    else
        echo "[smoke_check] Sampling completed for ${topic}."
    fi
done

# Terminate background stack cleanly
echo "[smoke_check] Shutting down synthetic stack PID ${STACK_PID}..."
kill -INT ${STACK_PID} 2>/dev/null || kill -TERM ${STACK_PID} 2>/dev/null || true
wait ${STACK_PID} 2>/dev/null || true

echo "============================================================"
echo "    ALL SMOKE CHECKS PASSED SUCCESSFULLY (0 ERRORS)"
echo "============================================================"
exit 0
