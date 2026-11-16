#!/bin/bash
# run_experiment1.sh — 실험 1: 그냥 끝까지 한 바퀴 돌려보기
#
# 목적
#   이상 탐지도 판정도 빼고, 10 라운드가 집계까지 끊김 없이 도는지만 본다.
#   라운드마다 사업장당 50 윈도우로 고정해 수집 편차를 없앤다.
#
# 평소 실행과 다른 점
#   FL_SENSOR_MAX_WINDOWS_PER_SITE=50   (상한)
#       기본 600 이다. 첫 라운드가 그때까지 쌓인 걸 전부 가져가서
#       mn1 467 / mn2 233 / mn3 232 처럼 들쭉날쭉했다. 50 으로 묶는다.
#
#   FL_SENSOR_MIN_WINDOWS=50            (하한)
#       기본 40 이다. 상한만 걸면 40개 모인 순간 깨어나 n=44 로 출발한다.
#       50 으로 올려 50개가 실제로 쌓일 때까지 기다리게 한다.
#       상한과 같은 값이어야 라운드마다 정확히 50개가 된다.
#
#   FL_SENSOR_ROUND_WAIT_SEC=300
#       하한을 못 채우면 포기하고 있는 것으로 진행하는 제한 시간.
#       기본 90초는 50개를 모으기에 짧을 수 있어 늘린다.
#
#   FL_VERDICT=0
#       판정을 끈다. 초반 라운드는 모델이 수렴 전이라 임계값이 흔들리고,
#       정상 데이터에서도 판정이 나온다. 지금 보려는 것과 무관하다.
#
#   FL_GLOBAL_ROUNDS=10
#       기본값과 같지만 실험 조건이므로 명시한다.
#
# 이상 주입(--anomaly)은 원래 run_fl.sh 가 안 넘긴다. 따로 끌 것이 없다.

set -euo pipefail
cd "$(dirname "$0")"

export FL_SENSOR_MAX_WINDOWS_PER_SITE="${FL_SENSOR_MAX_WINDOWS_PER_SITE:-50}"
export FL_SENSOR_MIN_WINDOWS="${FL_SENSOR_MIN_WINDOWS:-50}"
export FL_SENSOR_ROUND_WAIT_SEC="${FL_SENSOR_ROUND_WAIT_SEC:-300}"
export FL_VERDICT="${FL_VERDICT:-0}"
export FL_GLOBAL_ROUNDS="${FL_GLOBAL_ROUNDS:-10}"

echo "============================================================"
echo "  실험 1 — 이상 탐지 없이 10 라운드"
echo "============================================================"
echo "  사업장당 윈도우/라운드 : $FL_SENSOR_MIN_WINDOWS (쌓일 때까지 대기) ~ $FL_SENSOR_MAX_WINDOWS_PER_SITE"
echo "  수집 대기 제한         : ${FL_SENSOR_ROUND_WAIT_SEC}s"
echo "  판정                   : 끔 (FL_VERDICT=$FL_VERDICT)"
echo "  라운드                 : $FL_GLOBAL_ROUNDS"
echo "============================================================"

exec bash run_fl.sh "$@"
