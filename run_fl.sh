#!/bin/bash
# run_fl.sh — edgeway 연합학습 데모 기동
#
# 전제
#   1) MEC 샌드박스 시나리오(edgeway)가 떠 있다. MEP 3개 + IN-CSE.
#   2) Isaac Sim이 각 MN-CSE에 로봇 AE를 등록하고 telemetry를 보내고 있다.
#      로봇 AE는 우리가 만들지 않는다.
#   3) FL_SANDBOX_HOST / FL_SANDBOX_NAME 이 맞게 잡혀 있다.
#
# 오프라인으로 학습 로직만 돌려볼 때
#   FL_SENSOR_FROM_CSE=0 ./run_fl.sh
#   (CSE 대신 prepare_data_amr.py가 만든 pkl을 라운드별로 잘라 쓴다)

set -euo pipefail

cd "$(dirname "$0")"

PIDS=()
cleanup() {
  if [ ${#PIDS[@]} -gt 0 ]; then
    kill "${PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

FROM_CSE="${FL_SENSOR_FROM_CSE:-1}"

if [ "$FROM_CSE" = "0" ]; then
  echo "[1/4] 오프라인 모드: pkl 준비"
  echo "      AMR_DATA 가 가리키는 telemetry.json 들을 읽는다"
  python3 fl/prepare_data_amr.py
else
  echo "[1/4] oneM2M 리소스 확인"
  # 로봇 AE가 없으면 여기서 경고가 난다. Isaac을 먼저 띄워야 한다.
  if ! python3 fl/setup_resources_standard.py --verify; then
    echo
    echo "  리소스가 빠져 있다. 먼저 생성한다."
    python3 fl/setup_resources_standard.py
  fi
fi

echo
echo "[2/4] MN-AE 기동 (사업장 3곳)"
python3 -u fl/mn_ae_standard.py 0 5001 &
PIDS+=("$!")
echo "  MN-AE-1 (siteA / mep-a) PID: ${PIDS[-1]}"

python3 -u fl/mn_ae_standard.py 1 5002 &
PIDS+=("$!")
echo "  MN-AE-2 (siteB / mep-b) PID: ${PIDS[-1]}"

python3 -u fl/mn_ae_standard.py 2 5003 &
PIDS+=("$!")
echo "  MN-AE-3 (siteC / mep-c) PID: ${PIDS[-1]}"

# 각 MN-AE가 cnt-fl-control subscription을 만들 시간을 준다.
# IN-AE가 먼저 라운드 명령을 올리면 NOTIFY를 아무도 못 받는다.
sleep 5

echo
echo "[3/4] IN-AE 기동 (클라우드 IN-CSE 코디네이터)"
python3 -u fl/in_ae_standard.py &
PIDS+=("$!")
echo "  IN-AE PID: ${PIDS[-1]}"

sleep 2

echo
echo "[4/4] 대시보드"
echo "  -> http://localhost:7000"
python3 fl/dashboard_server.py
