#!/bin/bash
# run_fl.sh — 세 부분을 한 호스트에서 한꺼번에 띄운다 (edgeway-FL 저장소 전용 편의 스크립트)
#
# 실제 배치(mostonar/edgeway)에서는 호스트마다 따로 띄운다.
#   Cloud 호스트      Cloud/FL-FedAvg/run_fedavg.sh, Cloud/FL-Dashboard/run_dashboard.sh
#   Edge-ESTIMED-VM   Edge-ESTIMED-VM/Local-Model/run_local_model.sh
#
# 앞에 붙인 환경변수는 세 스크립트에 그대로 넘어간다.
#   예) FL_GLOBAL_ROUNDS=5 JUDGE_SITES=C ./run_fl.sh
#       bash run_fl.sh 2>&1 | tee /tmp/fl.log

set -uo pipefail
cd "$(dirname "$0")"

PIDS=()
cleanup() {
  [ ${#PIDS[@]} -gt 0 ] && kill "${PIDS[@]}" 2>/dev/null
  wait 2>/dev/null
}
trap cleanup EXIT INT TERM

bash Cloud/FL-Dashboard/run_dashboard.sh &
PIDS+=("$!")
sleep 3
bash Edge-ESTIMED-VM/Local-Model/run_local_model.sh &
PIDS+=("$!")
bash Cloud/FL-FedAvg/run_fedavg.sh &
PIDS+=("$!")
wait
