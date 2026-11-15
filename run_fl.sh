#!/bin/bash
# run_fl.sh — edgeway 연합학습 데모 기동
#
# 전제
#   1) MEC 샌드박스 시나리오(edgeway)가 떠 있다. MEP 3개 + 클라우드 IN-CSE.
#   2) Isaac Sim이 각 MN-CSE에 로봇 AE를 등록하고 telemetry를 보내고 있다.
#      로봇 AE는 우리가 만들지 않는다.
#   3) .env.local 에 샌드박스 주소가 들어 있고 source 되어 있다.
#
# 순서
#   리소스 확인/생성 → 대시보드 → Isaac 대기 → MN-AE → IN-AE
#
#   대시보드를 Isaac 대기보다 먼저 띄운다. 기다리는 동안 상태를 볼 수
#   있어야 하기 때문이다. 마지막에 두면 로봇이 안 올라올 때 몇 분간
#   아무것도 못 본다.
#
#   MN-AE를 IN-AE보다 먼저 띄운다. MN-AE가 cnt-fl-control 구독을 걸기 전에
#   IN-AE가 라운드 명령을 올리면 NOTIFY를 아무도 받지 못한다.
#
# 오프라인으로 학습 로직만 돌려볼 때
#   FL_SENSOR_FROM_CSE=0 ./run_fl.sh

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

start_dashboard() {
  echo
  echo "[2/5] 대시보드"
  python3 -u fl/dashboard_server.py &
  PIDS+=("$!")
  echo "  -> http://localhost:7000   (PID ${PIDS[-1]})"
  sleep 2
}

# ── 1. 데이터 경로 준비 ─────────────────────────────────

if [ "$FROM_CSE" = "0" ]; then
  echo "[1/5] 오프라인 모드: pkl 준비"
  echo "      AMR_DATA 가 가리키는 telemetry.json 들을 읽는다"
  python3 fl/prepare_data_amr.py

  start_dashboard

  echo
  echo "[3/5] Isaac Sim 대기 — 오프라인 모드라 건너뛴다"
else
  echo "[1/5] oneM2M 리소스 확인"
  # 우리 리소스(IN-AE, MN-AE-i, 컨테이너, ACP)만 본다.
  # 로봇 AE는 Isaac 소유라 여기서 실패로 치지 않는다.
  if ! python3 fl/setup_resources_standard.py --verify; then
    echo
    echo "  리소스가 빠져 있다. 먼저 생성한다."
    python3 fl/setup_resources_standard.py
  fi

  start_dashboard

  echo
  echo "[3/5] Isaac Sim 대기"
  # 샌드박스가 뜬 직후에는 MN-CSE만 있고 로봇 AE는 없다. Isaac이 등록한다.
  # 사업장당 최소 FL_MIN_ROBOTS_PER_SITE 대(기본 1)만 준비되면 진행한다.
  echo "  (기다리는 동안 대시보드는 이미 떠 있다)"
  if ! python3 fl/setup_resources_standard.py --wait-robots; then
    echo
    echo "  ⚠ 로봇이 다 준비되지 않았다. 그래도 진행한다."
    echo "    (뒤늦게 올라온 로봇은 그 라운드부터 포함된다)"
  fi
fi

# ── 2. 학습 노드 ────────────────────────────────────────

echo
echo "[4/5] MN-AE 기동 (사업장 3곳)"
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
sleep 5

# ── 3. 코디네이터 ───────────────────────────────────────

echo
echo "[5/5] IN-AE 기동 (클라우드 IN-CSE 코디네이터)"
python3 -u fl/in_ae_standard.py &
PIDS+=("$!")
echo "  IN-AE PID: ${PIDS[-1]}"

echo
echo "============================================================"
echo "  전부 기동됨.   대시보드: http://localhost:7000"
echo "  Ctrl+C 로 전체 종료"
echo "============================================================"
echo

# 백그라운드 프로세스가 끝날 때까지 기다린다.
# trap 이 걸려 있어 Ctrl+C 하면 전부 정리된다.
wait
