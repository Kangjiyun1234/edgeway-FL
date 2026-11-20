#!/bin/bash

# ============================================================
# FL 데모 초기화
#
# 초기화 대상:
#   - 기존 Dashboard 프로세스
#   - 기존 IN-AE 프로세스
#   - 기존 MN-AE 프로세스
#   - TinyIoT DB와 oneM2M 리소스
#   - 글로벌 모델
#   - 로컬 모델
#   - MN 모델 캐시
#
# 유지 대상:
#   - 전처리된 AMR PKL 데이터
#
# 전처리 데이터가 없을 때만 별도 실행:
#   python3 Edge-ESTIMED-VM/Local-Model/site_app/prepare_data_amr.py
# ============================================================

set -euo pipefail

cd "$(dirname "$0")"

DATA_DIR="${FL_PKL_DIR:-/tmp/fl_data/amr}"
MODEL_BASE_DIR="${FL_MODEL_BASE_DIR:-/tmp/fl_models}"
RUN_MARKER="${FL_DEMO_RUN_MARKER:-${MODEL_BASE_DIR}/.demo_run_started}"


echo "============================================================"
echo "FL 데모 초기화"
echo "============================================================"


# ------------------------------------------------------------
# 1. 전처리 데이터 확인
# ------------------------------------------------------------

echo
echo "[1/6] 전처리 데이터 확인"

for node in mn1 mn2 mn3
do
    data_file="${DATA_DIR}/${node}.pkl"

    if [ ! -f "${data_file}" ]; then
        echo
        echo "오류: 전처리 데이터가 없습니다."
        echo "  ${data_file}"
        echo
        echo "다음 명령을 먼저 실행하세요."
        echo "  python3 Edge-ESTIMED-VM/Local-Model/site_app/prepare_data_amr.py"
        exit 1
    fi

    echo "  ✓ ${data_file}"
done


# ------------------------------------------------------------
# 2. 기존 FL 및 Dashboard 프로세스 종료
# ------------------------------------------------------------

echo
echo "[2/6] 기존 FL 및 Dashboard 프로세스 종료"

pkill -TERM -f "Cloud/FL-Dashboard/dashboard_server.py" 2>/dev/null || true
pkill -TERM -f "Cloud/FL-FedAvg/in_ae_standard.py" 2>/dev/null || true
pkill -TERM -f "Edge-ESTIMED-VM/Local-Model/site_app/mn_ae_standard.py" 2>/dev/null || true

sleep 2

# 정상적으로 종료되지 않은 프로세스가 있을 경우 강제 종료
pkill -KILL -f "Cloud/FL-Dashboard/dashboard_server.py" 2>/dev/null || true
pkill -KILL -f "Cloud/FL-FedAvg/in_ae_standard.py" 2>/dev/null || true
pkill -KILL -f "Edge-ESTIMED-VM/Local-Model/site_app/mn_ae_standard.py" 2>/dev/null || true

echo "  ✓ Dashboard 종료"
echo "  ✓ IN-AE 종료"
echo "  ✓ MN-AE 종료"


# ------------------------------------------------------------
# 3. 이전 모델과 캐시 삭제
# ------------------------------------------------------------

echo
echo "[3/6] 이전 모델 및 캐시 삭제"

rm -rf "${MODEL_BASE_DIR}/global"
rm -rf "${MODEL_BASE_DIR}/local"
rm -rf "${MODEL_BASE_DIR}/cache"
rm -f "${RUN_MARKER}"

mkdir -p "${MODEL_BASE_DIR}/global"
mkdir -p "${MODEL_BASE_DIR}/local"
mkdir -p "${MODEL_BASE_DIR}/cache"

echo "  ✓ ${MODEL_BASE_DIR}/global 초기화"
echo "  ✓ ${MODEL_BASE_DIR}/local 초기화"
echo "  ✓ ${MODEL_BASE_DIR}/cache 초기화"


# ------------------------------------------------------------
# 4. TinyIoT DB 초기화 및 oneM2M 리소스 재생성
# ------------------------------------------------------------

echo
echo "[4/6] TinyIoT DB 초기화 및 oneM2M 리소스 재생성"

python3 Cloud/FL-FedAvg/setup_resources_standard.py --clean


# ------------------------------------------------------------
# 5. 새 데모 실행 표식 생성
# ------------------------------------------------------------

echo
echo "[5/6] 새 데모 실행 표식 생성"

mkdir -p "$(dirname "${RUN_MARKER}")"
touch "${RUN_MARKER}"

echo "  ✓ ${RUN_MARKER}"


# ------------------------------------------------------------
# 6. 센서 데이터
# ------------------------------------------------------------
#
# 예전 구조에서는 여기서 data_generator.py가 cnt-sensor-data에
# 라운드별 메타데이터 CIN을 올렸다. 지금은 그 컨테이너가 없다.
# 텔레메트리는 Isaac Sim이 로봇 AE의 telemetry 컨테이너에 직접 올리고,
# MN-AE가 거기서 최근 윈도우를 읽는다. 초기화할 것이 없다.

echo
echo "[6/6] 센서 데이터 — Isaac Sim이 로봇 AE에 직접 올린다 (할 일 없음)"


echo
echo "============================================================"
echo "FL 데모 초기화 완료"
echo "============================================================"
echo "전처리된 PKL 데이터는 삭제하지 않았습니다."
echo "이제 IN-AE, MN-AE, Dashboard를 각각 실행하세요."
echo "============================================================"