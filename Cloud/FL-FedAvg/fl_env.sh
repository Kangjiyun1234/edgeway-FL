# fl_env.sh — 이 폴더의 실행 스크립트가 맨 앞에서 source 하는 준비 단계.
# 직접 실행하는 파일이 아니다. 같은 내용이 FL-FedAvg, FL-Dashboard, Local-Model
# 세 폴더에 하나씩 있다(폴더마다 따로 배포할 수 있게 일부러 복사해 둠).
#
# 하는 일
#   1) .env.local 을 찾아 읽는다. 이 폴더에 있으면 그것을, 없으면 저장소
#      루트(이 폴더의 두 단계 위)의 것을 쓴다. 이미 셸에 있는 값은 덮어쓰지
#      않는다(앞에 붙인 환경변수가 이긴다).
#   2) 포트 비우기, torch 확인 같은 공용 함수를 정의한다.

FL_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FL_ROOT="$(cd "$FL_HERE/../.." && pwd)"
export FL_HERE FL_ROOT

fl_load_env() {
  local f="$FL_HERE/.env.local"
  [ -f "$f" ] || f="$FL_ROOT/.env.local"
  if [ ! -f "$f" ]; then
    echo "  ⚠ .env.local 이 없다 ($FL_HERE 또는 $FL_ROOT). 주소가 기본값으로 간다."
    echo "    cp $FL_HERE/env.local.example .env.local  한 뒤 채울 것"
    return 0
  fi
  # 이미 셸에 있는 값은 지킨다. .env.local 의 export 줄을 한 줄씩 보고
  # 아직 없는 변수만 받아들인다.
  local line name
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    line="${line#"${line%%[![:space:]]*}"}"      # 앞 공백 제거
    line="${line#export }"
    case "$line" in
      [A-Za-z_]*=*) ;;
      *) continue ;;                               # 주석, 빈 줄
    esac
    name="${line%%=*}"
    if [ -z "${!name+x}" ]; then
      eval "export $line"
    fi
  done < "$f"
  echo "  ✓ .env.local 읽음 ($f)"
}

fl_pythonpath() {
  # 모듈이 있는 폴더를 import 경로 맨 앞에 둔다
  export PYTHONPATH="$1${PYTHONPATH:+:$PYTHONPATH}"
}

fl_require_torch() {
  if ! python3 -c "import torch" 2>/dev/null; then
    echo
    echo "  ✗ 지금 python3 에 torch 가 없다: $(command -v python3)"
    echo "    가상환경을 켜고 다시 실행할 것 (pip install -r $FL_HERE/requirements.txt)"
    exit 1
  fi
}

# 이름으로 먼저 내리고, 그래도 포트를 잡고 있는 것은 포트로 내린다.
#   fl_free_ports "패턴1 패턴2" "포트1 포트2"
fl_free_ports() {
  local pats="$1" ports="$2" killed=0 pat port pid
  for pat in $pats; do
    pkill -TERM -f "$pat" 2>/dev/null && killed=1
  done
  [ "$killed" = "1" ] && sleep 2
  for pat in $pats; do
    pkill -KILL -f "$pat" 2>/dev/null || true
  done
  for port in $ports; do
    pid="$(ss -ltnpH "sport = :${port}" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1 || true)"
    if [ -n "${pid:-}" ]; then
      echo "  · 포트 ${port} 를 PID ${pid} 가 잡고 있다 -> 내린다"
      kill -TERM "$pid" 2>/dev/null || true
      sleep 1
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
  echo "  ✓ 포트 정리: ${ports}"
}

fl_log_dir() {
  # 실행별 로그 폴더. 각 컴포넌트 폴더 아래 logs/ (git 제외)
  local d="$1/logs/run_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$d"
  echo "$d"
}
