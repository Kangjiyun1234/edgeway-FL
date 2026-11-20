#!/bin/bash
# Cloud/FL-FedAvg/run_fedavg.sh — FL 코디네이터(IN-AE) 기동
#
# 순서
#   지난 실행 정리 → IN-CSE 리소스 확인/생성 → IN-AE
#
# 전제
#   - 이 폴더나 저장소 루트의 .env.local 에 FL_IN_BASE_URL / FL_IN_CSE_NAME
#     (필요하면 FL_IN_USER / FL_IN_PASS) 가 있다.
#   - 대시보드(Cloud/FL-Dashboard)는 같은 호스트에서 돈다.
#     IN-AE 가 쓰는 /tmp/fl_models/global 과 실행 표식을 대시보드가 읽는다.
#
# Local-Model(사업장 앱 3개)과 어느 쪽을 먼저 켜도 된다. IN-AE 는 첫 라운드
# 전에 사업장 앱 3개의 cnt-fl-control 구독이 생길 때까지 기다린다
# (FL_COMMAND_SUBSCRIPTION_WAIT, 이 스크립트 기본 900초). Local-Model 은
# 로봇을 최대 300초 기다린 뒤에야 사업장 앱을 띄우므로 60초로는 모자라다.
#
# 자주 쓰는 설정
#   FL_GLOBAL_ROUNDS=10     라운드 수
#   FL_VERIFY_ACP=0         시작 때 ACP 점검을 끈다. 이 점검은 사업장 MN-CSE 에
#                           직접 붙으므로, 이 호스트에서 샌드박스가 안 닿으면 끈다.
#   FL_RESET_ON_START=0     지난 실행의 글로벌 모델을 지우지 않는다.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/fl_env.sh"
cd "$FL_ROOT"

fl_load_env
fl_pythonpath "$HERE"
fl_require_torch

export FL_COMMAND_SUBSCRIPTION_WAIT="${FL_COMMAND_SUBSCRIPTION_WAIT:-900}"

# 시작 때 ACP 점검은 사업장 MN-CSE 에 직접 붙는다. 이 호스트에 샌드박스 주소가
# 없으면 닿을 곳이 없어 재시도만 반복하므로 끈다. (점검은 진단용이라 학습과 무관)
if [ -z "${FL_SANDBOX_HOST:-}" ] && [ -z "${FL_VERIFY_ACP:-}" ]; then
  export FL_VERIFY_ACP=0
  echo "  · FL_SANDBOX_HOST 가 없어 시작 때 ACP 점검(FL_VERIFY_ACP)을 끈다"
fi

PIDS=()
cleanup() { [ ${#PIDS[@]} -gt 0 ] && kill "${PIDS[@]}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo
echo "[1/3] 지난 실행 정리"
fl_free_ports "Cloud/FL-FedAvg/in_ae_standard.py" "6000"
if [ "${FL_RESET_ON_START:-1}" = "1" ]; then
  BASE="${FL_MODEL_BASE_DIR:-/tmp/fl_models}"
  MARKER="${FL_DEMO_RUN_MARKER:-$BASE/.demo_run_started}"
  # 글로벌 모델만 지운다. 이걸 안 하면 대시보드가 지난 실행의
  # global_roundN.pt 를 주워 읽어 지난 판의 라운드가 그대로 보인다.
  rm -rf "$BASE/global"
  mkdir -p "$BASE/global" "$(dirname "$MARKER")"
  : > "$MARKER"
  echo "  ✓ 글로벌 모델 비움: $BASE/global"
  echo "  ✓ 실행 표식 갱신: $MARKER"
fi

echo
echo "[2/3] IN-CSE 리소스 확인"
if ! python3 Cloud/FL-FedAvg/setup_resources_standard.py --scope in --verify; then
  echo
  echo "  리소스가 빠져 있다. 먼저 생성한다."
  python3 Cloud/FL-FedAvg/setup_resources_standard.py --scope in
fi

echo
echo "[3/3] IN-AE (코디네이터)"
LOG_DIR="$(fl_log_dir "$HERE")"
echo "  로그: $LOG_DIR/in_ae.log"
echo "  사업장 앱 구독 대기 최대 ${FL_COMMAND_SUBSCRIPTION_WAIT}초"
python3 -u Cloud/FL-FedAvg/in_ae_standard.py 2>&1 | tee "$LOG_DIR/in_ae.log" &
PIDS+=("$!")
wait
