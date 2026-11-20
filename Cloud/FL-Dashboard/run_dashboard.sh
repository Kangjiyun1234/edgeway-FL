#!/bin/bash
# Cloud/FL-Dashboard/run_dashboard.sh — FL 대시보드 기동 (http://localhost:7000)
#
# 대시보드는 IN-CSE 만 읽는다. 그리고 같은 호스트의 IN-AE(Cloud/FL-FedAvg)가
# 쓰는 /tmp/fl_models/global 과 실행 표식을 읽으므로 그 호스트에서 띄운다.
# FedAvg 보다 먼저 띄워도 된다. 기다리는 동안 상태를 볼 수 있다.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/fl_env.sh"
cd "$FL_ROOT"

fl_load_env
fl_pythonpath "$HERE"
fl_require_torch

PORT="${FL_DASHBOARD_PORT:-7000}"
echo
echo "[1/2] 지난 실행 정리"
fl_free_ports "Cloud/FL-Dashboard/dashboard_server.py" "$PORT"

echo
echo "[2/2] 대시보드 -> http://localhost:${PORT}"
LOG_DIR="$(fl_log_dir "$HERE")"
echo "  로그: $LOG_DIR/dashboard.log"
python3 -u Cloud/FL-Dashboard/dashboard_server.py 2>&1 | tee "$LOG_DIR/dashboard.log"
