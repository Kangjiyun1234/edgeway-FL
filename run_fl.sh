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

# ------------------------------------------------------------
# 지난 실행 흔적 정리
#
# 이걸 안 하면 대시보드가 초기화되지 않는다. /tmp/fl_models/global 에
# 남은 지난 실행의 global_roundN.pt 를 주워 읽고, 실행 표식(run marker)이
# 갱신되지 않아 "이 표식보다 오래된 모델은 무시" 하는 걸름망도 안 걸린다.
# 그래서 새로 띄워도 지난 판의 라운드와 수치가 그대로 보인다.
#
# clean_fl.sh 는 쓰지 않는다. 그쪽은 전처리 PKL 이 있어야 시작하는데
# CSE 모드에는 그 파일이 없어서 바로 멈춘다. 또 프로세스를 죽이는 일까지
# 하므로 기동 스크립트 안에서 부를 것이 아니다.
#
# 지우는 것: 지난 실행의 글로벌/로컬 모델과 캐시
# 남기는 것: 전처리 PKL (FL_PKL_DIR). 오프라인 모드에서 쓴다.
# ------------------------------------------------------------
reset_previous_run() {
  local base="${FL_MODEL_BASE_DIR:-/tmp/fl_models}"
  local marker="${FL_DEMO_RUN_MARKER:-${base}/.demo_run_started}"

  echo
  echo "[0/5] 지난 실행 정리"

  # 포트를 잡고 있는 지난 실행을 먼저 내린다.
  # 이걸 안 하면 7000(대시보드) 6000(IN-AE) 5001-5003(MN-AE) 이
  # 전부 "Address already in use" 로 bind 실패한다. 그러면 대시보드는
  # 뜨지 않고, MN-AE 는 구독은 걸지만 알림 받을 서버가 없어 폴링만
  # 하며, 옛 프로세스가 살아 있는 채로 같은 드롭박스에 같이 올려
  # 라운드가 어긋난다.
  #
  # 우리가 띄운 것만 이름으로 지목해 죽인다. 이 스크립트가 띄울
  # 자식들은 아직 없으므로 자기 자신을 죽이지 않는다.
  local killed=0
  for pat in "fl/dashboard_server.py" "fl/in_ae_standard.py" "fl/mn_ae_standard.py"
  do
    if pkill -TERM -f "$pat" 2>/dev/null; then
      killed=1
    fi
  done

  if [ "$killed" = "1" ]; then
    sleep 2
    for pat in "fl/dashboard_server.py" "fl/in_ae_standard.py" "fl/mn_ae_standard.py"
    do
      pkill -KILL -f "$pat" 2>/dev/null || true
    done
    echo "  ✓ 지난 실행 프로세스 종료 (이름 기준)"
  fi

  # 이름으로 못 잡는 것이 남는다. 다른 경로에서 띄웠거나, 부모가 먼저
  # 죽어 고아가 된 프로세스는 cmdline 이 달라 pkill -f 를 피해 간다.
  # 그런데 우리에게 필요한 건 "그 포트가 비어 있는 것" 하나뿐이다.
  # 그러니 이름이 아니라 포트를 기준으로 비운다.
  local busy=0
  for port in 7000 6000 5001 5002 5003
  do
    local pid
    pid="$(ss -ltnpH "sport = :${port}" 2>/dev/null \
           | grep -oP 'pid=\K[0-9]+' | head -1 || true)"
    if [ -n "${pid:-}" ]; then
      echo "  · 포트 ${port} 를 PID ${pid} 가 잡고 있다 -> 내린다"
      kill -TERM "$pid" 2>/dev/null || true
      busy=1
    fi
  done

  if [ "$busy" = "1" ]; then
    sleep 2
    for port in 7000 6000 5001 5002 5003
    do
      local pid
      pid="$(ss -ltnpH "sport = :${port}" 2>/dev/null \
             | grep -oP 'pid=\K[0-9]+' | head -1 || true)"
      [ -n "${pid:-}" ] && kill -KILL "$pid" 2>/dev/null || true
    done
  fi

  if [ "$killed" = "0" ] && [ "$busy" = "0" ]; then
    echo "  ✓ 남아 있는 프로세스 없음"
  else
    echo "  ✓ 포트 7000/6000/5001-5003 비움"
  fi

  rm -rf "${base}/global" "${base}/local" "${base}/cache"
  mkdir -p "${base}/global" "${base}/local" "${base}/cache"

  mkdir -p "$(dirname "${marker}")"
  : > "${marker}"

  echo "  ✓ 모델/캐시 비움: ${base}"
  echo "  ✓ 실행 표식 갱신: ${marker}"
}

if [ "${FL_RESET_ON_START:-1}" = "1" ]; then
  reset_previous_run
fi

start_dashboard() {
  echo
  echo "[2/5] 대시보드"
  python3 -u fl/dashboard_server.py &
  PIDS+=("$!")
  local pid="${PIDS[-1]}"
  echo "  -> http://localhost:7000   (PID ${pid})"

  # 띄워놓고 확인을 안 했다. 그래서 bind 에 실패해 죽어도 이 줄만
  # 찍히고 넘어갔고, 왜 대시보드가 안 뜨는지 매번 로그를 뒤져야 했다.
  # 최대 10초 기다리며 실제로 응답하는지 본다.
  local ok=0
  for _ in 1 2 3 4 5 6 7 8 9 10
  do
    sleep 1
    if ! kill -0 "$pid" 2>/dev/null; then
      break            # 프로세스가 죽었다. 더 기다릴 것 없다.
    fi
    if curl -sS -o /dev/null --max-time 2 "http://127.0.0.1:7000/" 2>/dev/null; then
      ok=1
      break
    fi
  done

  if [ "$ok" = "1" ]; then
    echo "  ✓ 대시보드 응답 확인"
  else
    echo
    echo "  ✗ 대시보드가 뜨지 않았다."
    if kill -0 "$pid" 2>/dev/null; then
      echo "    프로세스는 살아 있는데 7000 이 응답하지 않는다."
    else
      echo "    프로세스가 죽었다. 바로 위에 찍힌 파이썬 오류를 보라."
      echo "    'Address already in use' 면 7000 을 다른 것이 잡고 있다:"
      echo "      ss -ltnp | grep 7000"
    fi
    echo "    FL 자체는 계속 진행한다."
    echo
  fi
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
echo "[5/5] IN-AE"
# IN-AE 는 중앙 조정 앱이라 IN 노드에서 돈다. 설계서가 IN 노드를
# "중앙 인프라에서 CSE 를 운영하고 AE 도 실행할 수 있는 노드" 로 정의한다.
# 사업장 서버에서 띄우면 중앙 CSE 가 방화벽 뒤를 호출하지 못해
# <pollingChannel> 로 우회해야 했다. 중앙에서 돌면 직접 알림을 받는다.
#
# 중앙 CSE 옆에 우리 프로세스를 둘 자리가 없어서 여기서 띄운다.
# 공유 IN-CSE 는 엔드포인트만 제공되고, ESTIMED 호스팅 샌드박스도
# 프런트엔드 접근만 허용된다("not recommended for full development").
# 그래서 IN-AE 가 캠퍼스 방화벽 뒤에 놓이고, 중앙 CSE 가 우리를 호출하지
# 못한다. TS-0001 이 요청 도달 불가 AE 를 위해 정의한 <pollingChannel> 로
# 받는다. FL_NOTIFY_REMOTE 기본값이 pch 라 따로 줄 것은 없다.
#
# 나중에 중앙 노드에 자리가 생기면 FL_START_IN_AE=0 으로 끄고 거기서 띄운다.
if [ "${FL_START_IN_AE:-1}" = "1" ]; then
  python3 -u fl/in_ae_standard.py &
  PIDS+=("$!")
  echo "  IN-AE PID: ${PIDS[-1]}  (중앙 CSE 는 원격, 알림은 pollingChannel)"
else
  echo "  → 여기서는 안 띄운다 (FL_START_IN_AE=0). 중앙 노드에서 띄울 것."
fi

echo
echo "============================================================"
echo "  전부 기동됨.   대시보드: http://localhost:7000"
echo "  Ctrl+C 로 전체 종료"
echo "============================================================"
echo

# 백그라운드 프로세스가 끝날 때까지 기다린다.
# trap 이 걸려 있어 Ctrl+C 하면 전부 정리된다.
wait
