#!/bin/bash
# Edge-ESTIMED-VM/Local-Model/run_local_model.sh — 사업장 앱 3개 + 판정 앱 기동
#
# 순서
#   지난 실행 정리 → MN-CSE 리소스 확인/생성 → Isaac Sim 로봇 대기
#   → 사업장 앱 3개 (TRAINING-AE-i + ANALYSIS-AE-i) → 판정 앱 (JUDGE-AE-n)
#
# 전제
#   - MEC Sandbox `edgeway` 시나리오가 떠 있다 (MEP마다 MN-CSE 하나).
#   - 이 폴더나 저장소 루트의 .env.local 에 FL_SANDBOX_HOST / FL_SANDBOX_NAME,
#     FL_IN_BASE_URL / FL_IN_CSE_NAME, FL_NOTIFY_HOST 가 있다.
#     FL_NOTIFY_HOST:5001-5003 은 사업장 MN-CSE 가 알림을 보내는 주소라
#     샌드박스에서 닿아야 한다.
#
# Cloud/FL-FedAvg 와 어느 쪽을 먼저 켜도 된다. IN-AE 가 이쪽 구독을 기다린다.
#
# 자주 쓰는 설정
#   JUDGE_SITES=A,B,C      띄울 판정 앱 (기본 A,B,C). 비우면 판정 앱 없이
#   JUDGE_AE_DIR=…         judge_ae.py 가 있는 폴더 (기본: 이 폴더의 judge_ae/)
#   FL_SENSOR_FROM_CSE=0   오프라인: AMR_DATA 의 telemetry 로 pkl 을 만들어 학습
#   FL_RESET_ON_START=0    지난 실행의 로컬 모델·캐시를 지우지 않는다

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/fl_env.sh"
cd "$FL_ROOT"

fl_load_env
fl_pythonpath "$HERE/site_app"
fl_require_torch

SITE_APP="Edge-ESTIMED-VM/Local-Model/site_app"
FROM_CSE="${FL_SENSOR_FROM_CSE:-1}"
JUDGE_SITES="${JUDGE_SITES-A,B,C}"
JUDGE_AE_DIR="${JUDGE_AE_DIR:-$HERE/judge_ae}"

PIDS=()
cleanup() { [ ${#PIDS[@]} -gt 0 ] && kill "${PIDS[@]}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo
echo "[1/5] 지난 실행 정리"
fl_free_ports "$SITE_APP/mn_ae_standard.py judge_ae.py" "5001 5002 5003"
if [ "${FL_RESET_ON_START:-1}" = "1" ]; then
  BASE="${FL_MODEL_BASE_DIR:-/tmp/fl_models}"
  rm -rf "$BASE/local" "$BASE/cache"
  mkdir -p "$BASE/local" "$BASE/cache"
  echo "  ✓ 로컬 모델/캐시 비움: $BASE"
fi

if [ "$FROM_CSE" = "0" ]; then
  echo
  echo "[2/5] 오프라인 모드: pkl 준비 (AMR_DATA 의 telemetry.json)"
  python3 "$SITE_APP/prepare_data_amr.py"
  echo
  echo "[3/5] 로봇 대기 — 오프라인 모드라 건너뛴다"
else
  echo
  echo "[2/5] MN-CSE 리소스 확인"
  # 우리 리소스(TRAINING-AE-i, ANALYSIS-AE-i, 컨테이너, ACP)만 본다.
  # 로봇 AE 는 Isaac 소유라 없어도 실패로 치지 않는다.
  if ! python3 "$SITE_APP/setup_resources_standard.py" --scope mn --verify; then
    echo
    echo "  리소스가 빠져 있다. 먼저 생성한다."
    python3 "$SITE_APP/setup_resources_standard.py" --scope mn
  fi

  echo
  echo "[3/5] Isaac Sim 로봇 대기 (사업장당 최소 ${FL_MIN_ROBOTS_PER_SITE:-1}대)"
  if ! python3 "$SITE_APP/setup_resources_standard.py" --scope mn --wait-robots; then
    echo
    echo "  ⚠ 로봇이 다 준비되지 않았다. 그래도 진행한다."
    echo "    (뒤늦게 올라온 로봇은 그 라운드부터 포함된다)"
  fi
fi

echo
echo "[4/5] 사업장 앱 3개"
LOG_DIR="$(fl_log_dir "$HERE")"
echo "  로그: $LOG_DIR/"
for i in 0 1 2; do
  n=$((i + 1))
  port=$((5001 + i))
  python3 -u "$SITE_APP/mn_ae_standard.py" "$i" "$port" 2>&1 | tee "$LOG_DIR/mn${n}.log" &
  PIDS+=("$!")
  echo "  TRAINING-AE-${n} (port ${port}) PID: ${PIDS[-1]}"
done

echo
echo "[5/5] 판정 앱"
if [ "$FROM_CSE" = "0" ]; then
  echo "  오프라인 모드라 띄우지 않는다."
elif [ -z "$JUDGE_SITES" ]; then
  echo "  JUDGE_SITES 가 비어 있어 띄우지 않는다."
elif [ ! -f "$JUDGE_AE_DIR/judge_ae.py" ]; then
  echo "  ⚠ $JUDGE_AE_DIR/judge_ae.py 가 없다. 판정 앱 없이 진행한다."
  echo "    mostonar/edgeway 의 Edge-ESTIMED-VM/Local-Model/judge_ae 를 JUDGE_AE_DIR 로 지정할 것"
else
  INSECURE=""
  case "${FL_TLS_VERIFY:-1}" in 0|false|False) INSECURE="--insecure" ;; esac
  for site in ${JUDGE_SITES//,/ }; do
    # 사업장 앱과 같은 규칙으로 CSE 주소를 만든다 (FL_MN_BASE_URL_MNx 등 덮어쓰기 존중).
    read -r CSE_URL IOT_URL < <(python3 - "$site" <<'PY'
import sys
import config, mec_iot
node = {"A": "mn1", "B": "mn2", "C": "mn3"}[sys.argv[1].upper()]
base, cse = config.mn_endpoint(node)
print(f"{base}/{cse}", f"{mec_iot.iot_base(config.MEP_OF_NODE[node])}/registered_iot_platforms")
PY
)
    python3 -u "$JUDGE_AE_DIR/judge_ae.py" --site "$site" --cse "$CSE_URL" --iot "$IOT_URL" $INSECURE \
      2>&1 | tee "$LOG_DIR/judge_${site}.log" &
    PIDS+=("$!")
    echo "  JUDGE-AE (site ${site}) PID: ${PIDS[-1]}  -> $CSE_URL"
  done
fi

echo
echo "============================================================"
echo "  전부 기동됨. 로그: $LOG_DIR/"
echo "  Ctrl+C 로 전체 종료"
echo "============================================================"
wait
