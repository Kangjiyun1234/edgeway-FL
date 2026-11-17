"""
fl/config.py — FL + oneM2M 통합 설정 (AMR 바닥·로봇 점검 서비스)

- 사업장 3곳
  - mn1 = 사업장 A : 로봇 4대, 학습 참여
  - mn2 = 사업장 B : 로봇 2대, 학습 참여
  - mn3 = 사업장 C : 로봇 4대, 공통 모델을 받아 자기 데이터로만 보정.
                     FedAvg에는 기여하지 않는다 (FL_LOCAL_ONLY_NODES)
- 조건부 1D CNN VAE(model_cvae.CVAE1D) 기반 이상 탐지
- 정상 주행 데이터만으로 학습한다

[CSE 구성]
사업장 CSE(MN-CSE)와 중앙 CSE(IN-CSE)가 분리되어 있다.
MN-CSE는 현장 MEC 호스트에, IN-CSE는 클라우드에 있다.
AE는 Isaac Sim 쪽에서 등록하므로 이 코드는 AE를 만들지 않는다.
"""

import os
from dataclasses import dataclass


# ════════════════════════════════════════════════════════
# AE 모델 설정
# ════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AEConfig:
    """조건부 1D CNN VAE 규격. prepare_data_amr.py 출력과 맞춰야 한다."""

    # 60Hz × 1초 = 윈도우 한 개
    seq_len: int = 60

    # IMU 24채널 + 엔코더 14채널
    n_channels: int = 38

    # 조건 벡터: nav 그룹(5) + 속도 버킷(3). amr_schema.COND_DIM 과 같아야 한다.
    cond_dim: int = 8

    # 잠재 차원과 은닉 채널. 이 조합에서 90,806 파라미터이고
    # fp16+gzip+base64로 약 217KB라 contentInstance 하나에 들어간다.
    latent_dim: int = 24
    hidden: int = 48

    # ELBO의 KL 가중치
    beta: float = 0.5

    # 이상 점수 집계: 0이면 전 채널 평균, n>0이면 상위 n개 채널 평균
    score_topk: int = 0


# ════════════════════════════════════════════════════════
# 학습 설정
# ════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TrainConfig:
    rounds: int = 10
    local_epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"
    seed: int = 42
    log_every: int = 1

    # 레거시 필드
    train_split: float = 0.8
    hidden_size: int = 128
    num_classes: int = 2


# ════════════════════════════════════════════════════════
# 데이터 설정
# ════════════════════════════════════════════════════════

# prepare_data_amr.py 가 만드는 PKL 디렉터리.
AMR_DATA_DIR = os.getenv(
    "FL_PKL_DIR",
    "/tmp/fl_data/amr",
)

# 하위 호환용 별칭 (옛 코드가 참조할 수 있다)


NODE_DATA_FILES = {
    "mn1": os.path.join(AMR_DATA_DIR, "mn1.pkl"),
    "mn2": os.path.join(AMR_DATA_DIR, "mn2.pkl"),
    "mn3": os.path.join(AMR_DATA_DIR, "mn3.pkl"),
}

# 사업장 이름과 로봇 대수 (prepare_data_amr.py 와 일치해야 한다)
SITE_OF_NODE = {"mn1": "siteA", "mn2": "siteB", "mn3": "siteC"}
ROBOTS_PER_NODE = {"mn1": 4, "mn2": 2, "mn3": 2}

# Isaac Sim이 로봇 AE 아래에 만드는 컨테이너 이름.
# 센서 데이터는 MN-AE가 아니라 로봇 AE 아래에 쌓인다.
#   <MN-CSE>/siteA_AMR_1/telemetry/<cin>
ROBOT_TELEMETRY_CNT = os.getenv("FL_ROBOT_TELEMETRY_CNT", "telemetry")
ROBOT_STATUS_CNT = os.getenv("FL_ROBOT_STATUS_CNT", "status")


def robot_names(node: str):
    """한 사업장의 로봇 AE 이름 목록. siteA_AMR_1 … siteA_AMR_4"""
    site = SITE_OF_NODE.get(node, node)
    return [
        f"{site}_AMR_{i}"
        for i in range(1, ROBOTS_PER_NODE.get(node, 0) + 1)
    ]

CLASS_NAMES = ["정상", "이상"]
NUM_CLASSES = 2

# RMS가 처음 이 값을 초과한 파일을 fault onset으로 사용
RMS_FAULT_THRESHOLD = 1.0


# ════════════════════════════════════════════════════════
# Anomaly Detection 설정
# ════════════════════════════════════════════════════════

# 재구성 오차가 threshold를 연속 K회 초과하면 fault 판정
ANOMALY_K_CONSECUTIVE = 3

# threshold = 중앙값 + N x MAD x 1.4826  (평균+Nσ 가 아니다)
#
# 지금까지 edge_node.calibrate() 가 이 값을 안 쓰고 자기 기본값 3.0 을
# 썼다. 그래서 이걸 고쳐도 실제 임계값이 안 바뀌었다. 연결했다.
#
# 기본값이 조건별 정규화(FL_COND_NORM) 여부에 따라 다르다. 정규화를 하면
# 학습 점수의 중앙값이 1 로 모이면서 MAD 가 같이 줄어든다. 3 을 그대로 쓰면
# 임계값이 내려가 오경보가 오히려 는다. 실측(1사업장, 평가 318정상/133이상):
#
#   정규화 없음  3MAD   임계 5.52   탐지 99.2%   오경보 48.7%
#   정규화 후    3MAD   임계 3.58   탐지 95.5%   오경보 71.4%   <- 더 나쁨
#   정규화 후    8MAD   임계 7.87   탐지 90.2%   오경보 38.1%
#   정규화 후   12MAD   임계 11.31  탐지 82.7%   오경보 28.0%
#
# 정규화 없이 n_sigma 만 8 로 올렸을 때가 탐지 75.9% / 오경보 31.1% 였으므로,
# 정규화 후 8 은 탐지와 오경보 양쪽에서 그보다 낫다. 그래서 8 을 기본으로
# 둔다. 탐지를 더 원하면 5, 오경보를 더 줄이려면 12 로 바꾸면 된다.
COND_NORM = os.getenv("FL_COND_NORM", "1") not in ("0", "false", "False")
THRESHOLD_N_SIGMA = float(
    os.getenv("FL_THRESHOLD_N_SIGMA", "8.0" if COND_NORM else "3.0"))

# 데모에서 이상 상태를 표시할 노드
ANOMALY_DEMO_NODE = os.getenv(
    "FL_ANOMALY_DEMO_NODE",
    "mn3",
)

# MN3는 기본적으로 Round 7부터 이상 샘플을 표시한다.
ANOMALY_START_ROUND = int(
    os.getenv(
        "FL_ANOMALY_START_ROUND",
        "7",
    )
)


# ════════════════════════════════════════════════════════
# oneM2M 연결 설정
# ════════════════════════════════════════════════════════

# ETSI MEC Sandbox의 ACME CSE를 쓴다.
#
# 사업장마다 MEP가 따로 있고 MN-CSE도 따로다.
#   mn1 = 사업장 A = mep-a
#   mn2 = 사업장 B = mep-b
#   mn3 = 사업장 C = mep-c
#
# 실제 주소 형태:
#   https://<host>/<sandbox>/<mep>/meep-acme-mn-cse/acme-cse-mn-<mep>-<sandbox>
#   └───────── base url ─────────┘ └──────── CSE 이름 ────────┘
#
# 샌드박스 주소는 코드에 두지 않는다. 공개 저장소이므로 환경변수로 받는다.
#
#   export FL_SANDBOX_HOST=<샌드박스가 뜬 호스트>
#   export FL_SANDBOX_NAME=<샌드박스 인스턴스 이름>
#
# 어디서 실행하느냐에 따라 호스트가 달라진다.
#   유선망            연구실 공인 IP
#   같은 공유기 Wi-Fi  사설 IP (192.168.x.x)
#   테일넷            100.x.x.x  — Isaac Sim PC도 테일넷이라 보통 이쪽이 편하다
#
# .env.local 에 적어 두고 셸에서 읽어 들이면 된다. .gitignore가 .env* 를 막는다.
#
# MN-CSE는 REMOTE_CSE_* 설정으로 IN-CSE에 등록된다(Mcc).
# 사업장↔중앙 연결은 CSE가 처리하고, 이 코드는 각 CSE에 Mca로 붙는다.

SANDBOX_SCHEME = os.getenv("FL_SANDBOX_SCHEME", "https")
SANDBOX_HOST = os.getenv("FL_SANDBOX_HOST", "").strip()
SANDBOX_NAME = os.getenv("FL_SANDBOX_NAME", "").strip()

# 설정이 안 됐을 때 URL에 들어갈 자리표시자.
# .invalid 는 RFC 2606이 예약한 도메인이라 반드시 이름 해석에 실패한다.
# 빈 문자열을 그대로 쓰면 "https:///..." 같은 주소가 만들어져서
# 왜 실패했는지 알기 어렵다.
_HOST_UNSET = "sandbox-host-not-set.invalid"
_NAME_UNSET = "sandbox-name-not-set"


# CSE가 HTTP 기본 인증을 요구할 때 쓴다.
#
# 샌드박스의 MN-CSE는 인증 없이 열려 있지만, 클라우드 IN-CSE는 401을 돌려준다.
# 자격증명은 코드에 두지 않는다. 공개 저장소이므로 환경변수로만 받는다.
# edgeway.yaml 의 REMOTE_CSE_USER / REMOTE_CSE_PASS 와 같은 값이다.
IN_AUTH_USER = os.getenv("FL_IN_USER", "")
IN_AUTH_PASS = os.getenv("FL_IN_PASS", "")
MN_AUTH_USER = os.getenv("FL_MN_USER", "")
MN_AUTH_PASS = os.getenv("FL_MN_PASS", "")


def in_auth():
    """IN-CSE용 (user, pass). 설정이 없으면 None."""
    return (IN_AUTH_USER, IN_AUTH_PASS) if IN_AUTH_USER else None


def mn_auth():
    """MN-CSE용 (user, pass). 설정이 없으면 None."""
    return (MN_AUTH_USER, MN_AUTH_PASS) if MN_AUTH_USER else None


def sandbox_ready() -> bool:
    """샌드박스 주소가 설정돼 있는가."""
    return bool(SANDBOX_HOST and SANDBOX_NAME)


def require_sandbox() -> None:
    """
    샌드박스에 붙어야 하는 진입점에서 먼저 부른다.
    설정이 없으면 무엇을 해야 하는지 알리고 멈춘다.

    import 시점에 막지 않는 이유는, prepare_data_amr.py 처럼 파일만
    다루는 작업은 샌드박스 없이도 돌아가야 하기 때문이다.
    """
    if sandbox_ready():
        return
    raise SystemExit(
        "\n샌드박스 주소가 설정되지 않았다.\n"
        "  export FL_SANDBOX_HOST=<호스트>     # 예: 테일넷 주소나 공인 IP\n"
        "  export FL_SANDBOX_NAME=<인스턴스 이름>\n"
        "\n"
        "공개 저장소라 실제 주소를 코드에 두지 않는다.\n"
        ".env.local 에 적어 두고 `source .env.local` 해도 된다.\n"
    )

# 샌드박스가 IP 기반 자체 서명 인증서를 쓰면 검증이 실패한다.
# 그때만 0으로 내린다. 끄면 중간자 공격을 막지 못한다.
TLS_VERIFY = os.getenv("FL_TLS_VERIFY", "1") not in ("0", "false", "False")

MEP_OF_NODE = {"mn1": "mep-a", "mn2": "mep-b", "mn3": "mep-c"}


def mn_endpoint(node: str):
    """사업장 노드의 (base_url, cse_name)."""
    mep = MEP_OF_NODE.get(node, "mep-a")
    override_url = os.getenv(f"FL_MN_BASE_URL_{node.upper()}")
    override_cse = os.getenv(f"FL_MN_CSE_NAME_{node.upper()}")
    base = override_url or (
        f"{SANDBOX_SCHEME}://{SANDBOX_HOST or _HOST_UNSET}"
        f"/{SANDBOX_NAME or _NAME_UNSET}/{mep}/meep-acme-mn-cse"
    )
    cse = override_cse or f"acme-cse-mn-{mep}-{SANDBOX_NAME or _NAME_UNSET}"
    return base, cse


def in_endpoint():
    """중앙 CSE의 (base_url, cse_name)."""
    base = os.getenv("FL_IN_BASE_URL") or (
        f"{SANDBOX_SCHEME}://{SANDBOX_HOST or _HOST_UNSET}"
        f"/{SANDBOX_NAME or _NAME_UNSET}/meep-acme-in-cse"
    )
    cse = os.getenv("FL_IN_CSE_NAME") or f"acme-cse-in-{SANDBOX_NAME or _NAME_UNSET}"
    return base, cse


# 기본값(단일 엔드포인트를 가정하는 기존 코드용). 사업장 A를 가리킨다.
# MN-AE는 기동 시 자기 노드의 엔드포인트로 onem2m_utils.set_endpoint()를 호출한다.
MN_BASE_URL, MN_CSE_NAME = mn_endpoint("mn1")
MN_CSE_ID = os.getenv("FL_MN_CSE_ID", "")

IN_BASE_URL, IN_CSE_NAME = in_endpoint()
IN_CSE_ID = os.getenv("FL_IN_CSE_ID", "")

IN_IS_REMOTE = (IN_BASE_URL != MN_BASE_URL) or (IN_CSE_NAME != MN_CSE_NAME)

# 기존 코드가 참조하는 이름. 사업장 CSE를 가리킨다.
BASE_URL = MN_BASE_URL
CSE_NAME = MN_CSE_NAME

# 중앙 조정 앱의 AE 이름과 AE-ID.
#
# IN-CSE 는 여러 팀이 같이 쓰는 공유 자원이다. "IN-AE" 는 누구나 쓸 법한
# 이름이라 남의 것과 헷갈리거나 먼저 선점당할 수 있다. 바꾸려면 두 값을
# 같이 바꾸고 setup_resources_standard.py 를 다시 돌려야 한다.
# 컨테이너와 ACP 가 전부 이 이름 아래에 매달려 있기 때문이다.
IN_AE_NAME = os.getenv("FL_IN_AE_NAME", "IN-AE")

# AE는 Isaac Sim 쪽에서 등록한다. setup_resources는 만들지 않고 존재만 확인한다.
CREATE_AES = os.getenv("FL_CREATE_AES", "0") in ("1", "true", "True")

# AE가 올라올 때까지 기다릴 시간(초)
AE_WAIT_SEC = float(os.getenv("FL_AE_WAIT_SEC", "60"))

NOTIFY_HOST = os.getenv(
    "FL_NOTIFY_HOST",
    "127.0.0.1",
)

# 알림을 어떤 경로로 받을지.
#
#   "direct"  CSE가 우리 HTTP 서버를 직접 호출한다. 구독의 nu 는 URL 이고
#             AE 에 poa 가 박혀 있어야 한다. CSE가 AE에 닿을 수 있을 때.
#   "pch"     oneM2M <pollingChannel>(m2m:pch, ty=15)로 받는다. AE가 CSE에
#             길게 걸어 두고 CSE가 그 응답으로 알림을 내려보낸다. TS-0001이
#             요청 도달 불가(request-unreachable) AE를 위해 정의한 방식이다.
#
# 구독도 알림도 그대로다. 전달 방향만 뒤집힌다.
#
# [왜 하나가 아니라 둘인가]
# 전에는 플래그 하나로 모든 알림 경로를 정했다. 그런데 우리 배치는 경로마다
# 조건이 다르다.
#
#   로봇 → MN-CSE → MN-AE     사업장 내부. CSE가 우리를 부를 수 있다  → direct
#   IN-CSE → MN-AE / IN-AE    방화벽 너머. 부를 수 없다               → pch
#
# 한 플래그로 둘 다 정하면 원격 때문에 pch 를 켜는 순간 방화벽과 아무 상관없는
# 사업장 안의 텔레메트리 구독까지 같이 죽는다(poa 를 안 박게 되므로). 반대로
# 로컬 때문에 direct 로 두면 원격 구독이 전부 타임아웃이다.
#
# 조건이 다르면 프로토콜도 달라야 한다. 그래서 둘로 나눈다.
NOTIFY_LOCAL = os.getenv(
    "FL_NOTIFY_LOCAL", os.getenv("FL_NOTIFY_MODE", "direct")
).strip().lower()

NOTIFY_REMOTE = os.getenv(
    "FL_NOTIFY_REMOTE", os.getenv("FL_NOTIFY_MODE", "pch")
).strip().lower()


def local_notify_mode() -> str:
    """사업장 안 MN-CSE 로부터 받는 알림의 전달 방식."""
    return NOTIFY_LOCAL


def remote_notify_mode() -> str:
    """클라우드 IN-CSE 로부터 받는 알림의 전달 방식."""
    return NOTIFY_REMOTE


def remote_uses_pch() -> bool:
    return NOTIFY_REMOTE == "pch"

NUM_CLIENTS = 3
# 연합학습 라운드 수. 실험할 때 줄이거나 늘린다.
GLOBAL_ROUNDS = int(os.getenv("FL_GLOBAL_ROUNDS", "10"))

ORIGINATOR = os.getenv("FL_ORIGINATOR", "CAdmin")

# 각 AE가 자기 AE-ID 로 CSE에 요청할지.
#
# 꺼져 있으면 모두 ORIGINATOR(기본 CAdmin)로 나간다. 관리자 신원은 ACP를
# 전부 통과하므로 정책이 걸려 있어도 실제로는 아무것도 막지 못한다.
#
# 켜면 TRAINING-AE-1 은 CTRAINING-AE-1, IN-AE 는 CIN-AE 로 요청한다. 그래야 ACP가
# 실제로 작동하고 "사업장 간 격리"가 말뿐이 아니게 된다.
#
# 다만 정책에 구멍이 있으면 그 순간부터 동작이 막힌다. 로봇 AE 쪽 ACP가
# 들어온 뒤에 켜라. setup_resources_standard.py --verify 로 먼저 확인할 것.
USE_AE_IDENTITY = os.getenv("FL_USE_AE_IDENTITY", "0") == "1"

# AE-ID. ACP의 acor 와 X-M2M-Origin 에 쓰는 값이다.
# 한 곳에서 정해야 ACP와 실제 요청이 어긋나지 않는다.
IN_AE_ID = os.getenv("FL_IN_AE_ID", "CIN-AE")


def training_ae_name(node: str) -> str:
    """mn1 -> TRAINING-AE-1

    사업장의 학습 앱이다. 전에는 MN-AE-i 였다. 같은 앱이 MN-CSE 와 IN-CSE
    양쪽에 등록돼 있어서 IN-CSE 트리에 "MN" 이 튀어나왔고, 거기 사는 앱으로
    읽혔다. 분석 앱(ANALYSIS-AE-i)처럼 하는 일로 이름을 맞춘다.
    """
    return f"TRAINING-AE-{node[2:]}"


def training_ae_id(node: str) -> str:
    """mn1 -> CTRAINING-AE-1"""
    return f"C{training_ae_name(node)}"


# ── 분석 앱(판정) ────────────────────────────────────────
#
# 설계서에서 사업장 안의 앱은 분석 / 운행 / 학습으로 나뉜다. 학습 앱이
# TRAINING-AE-i 고, 판정은 분석 앱의 일이다. 지금은 모델을 들고 있는 학습 앱이
# 계산까지 하지만,
# 리소스는 분석 앱 신원으로 남긴다. 나중에 분석 앱을 별도 프로세스로 떼어내도
# 리소스 트리와 구독하는 쪽 코드가 바뀌지 않는다.
#
# 판정 결과는 사업장 밖으로 나가지 않으므로 MN-CSE 에 둔다.
ANALYSIS_AE_PREFIX = os.getenv("FL_ANALYSIS_AE_PREFIX", "ANALYSIS-AE")
VERDICT_CNT = os.getenv("FL_VERDICT_CNT", "cnt-verdict")
VERDICT_MNI = int(os.getenv("FL_VERDICT_MNI", "500"))
VERDICT_ENABLED = os.getenv("FL_VERDICT", "1") not in ("0", "false", "False")

# 이 라운드 전에는 판정을 내지 않는다.
#
# 초반 라운드의 임계값은 거의 학습되지 않은 모델로 잰 것이라 판정이
# 의미를 갖지 못한다. 고정 검증셋 측정에서 AUROC 가 R1 0.615 -> R3 0.683
# 까지 오르고 그 뒤로 평평했으므로 기본값을 3 으로 둔다.
VERDICT_START_ROUND = int(os.getenv("FL_VERDICT_START_ROUND", "3"))


def analysis_ae_name(node: str) -> str:
    """mn1 -> ANALYSIS-AE-1"""
    return f"{ANALYSIS_AE_PREFIX}-{node[2:]}"


def analysis_ae_id(node: str) -> str:
    """mn1 -> CANALYSIS-AE-1"""
    return f"C{analysis_ae_name(node)}"


# IN-AE 의 ACP 자가진단을 돌릴지.
#
# 이 진단은 사업장 MN-CSE 3곳에 직접 접속해서 권한을 확인한다. IN-AE 가
# 사업장과 같은 네트워크에 있을 때만 가능하다.
#
# 설계상 IN-AE 는 IN 노드(중앙)에서 돈다. 거기서는 사업장 MN-CSE 가 방화벽
# 뒤에 있어 닿지 않으므로 이 진단을 끈다. 끄는 것은 기능이 아니라 진단이므로
# 연합학습 동작에는 영향이 없다.
VERIFY_ACP = os.getenv("FL_VERIFY_ACP", "1") not in ("0", "false", "False")

# X-M2M-RI(Request Identifier)는 TS-0009 HTTP 바인딩에서 모든 요청에 필수다.
# 요청마다 고유값이어야 하므로 여기 고정값을 두지 않고
# onem2m_utils._headers() 에서 매번 생성한다.
# 공통 헤더. Content-Type은 넣지 않는다.
#
# TS-0009에서 Content-Type의 ty 파라미터는 CREATE에만 붙는다. RETRIEVE나
# DELETE처럼 본문이 없는 요청에 ty=4를 실어 보내면 CSE가 거부한다.
# 예전에는 여기에 ty=4를 기본값으로 박아 두어 모든 GET이 실패했다.
# 필요한 요청에만 onem2m_utils._headers(ty=...) 로 붙인다.
HEADERS = {
    "X-M2M-Origin": ORIGINATOR,
    "X-M2M-RVI": os.getenv("FL_RVI", "3"),
    "Accept": "application/json",
}

# contentInstance의 con에 JSON 문자열을 넣을 때 표시할 contentInfo
CIN_CONTENT_INFO = os.getenv("FL_CIN_CNF", "application/json:0")


# ════════════════════════════════════════════════════════
# 모델 저장 경로
# ════════════════════════════════════════════════════════

MODEL_BASE_DIR = os.getenv(
    "FL_MODEL_BASE_DIR",
    "/tmp/fl_models",
)

LOCAL_MODEL_DIR = os.path.join(
    MODEL_BASE_DIR,
    "local",
)

GLOBAL_MODEL_DIR = os.path.join(
    MODEL_BASE_DIR,
    "global",
)

CACHE_MODEL_DIR = os.path.join(
    MODEL_BASE_DIR,
    "cache",
)

# clean_fl.sh가 새로운 데모 실행마다 이 파일의 수정 시각을 갱신한다.
#
# Dashboard는 이 시각보다 오래된 global_round*.pt 파일을
# 이전 실행에서 남은 stale model로 판단하고 사용하지 않는다.
DEMO_RUN_MARKER = os.getenv(
    "FL_DEMO_RUN_MARKER",
    os.path.join(
        MODEL_BASE_DIR,
        ".demo_run_started",
    ),
)

os.makedirs(
    LOCAL_MODEL_DIR,
    exist_ok=True,
)

os.makedirs(
    GLOBAL_MODEL_DIR,
    exist_ok=True,
)

os.makedirs(
    CACHE_MODEL_DIR,
    exist_ok=True,
)


# ════════════════════════════════════════════════════════
# DP-SGD 설정
# ════════════════════════════════════════════════════════

DP_EPSILON = 12.0
DP_DELTA = 5e-4
DP_MAX_GRAD_NORM = 1.5


# ════════════════════════════════════════════════════════
# cold start 노드 설정
# ════════════════════════════════════════════════════════

# 노드 역할은 셋이다.
#
#   학습 참여      : 로컬 학습 + FedAvg 기여          (기본)
#   로컬 전용      : 로컬 학습은 하되 FedAvg 미기여    FL_LOCAL_ONLY_NODES
#   관측 전용      : 학습 안 하고 추론만              FL_COLD_START_NODES
#
# 사업장 C(mn3)는 '로컬 전용'이다. 공통 모델을 받아 자기 정상 데이터로
# 보정하면서 점점 안정화되지만, 가중치는 중앙에 올리지 않는다.
# 관측 전용은 자체 데이터도 쓰지 않는 더 엄격한 신규 사업장 시연용이다.
COLD_START_NODES = [
    n.strip()
    for n in os.getenv("FL_COLD_START_NODES", "").split(",")
    if n.strip()
]

LOCAL_ONLY_NODES = [
    n.strip()
    for n in os.getenv("FL_LOCAL_ONLY_NODES", "mn3").split(",")
    if n.strip()
]

ALL_NODES = [f"mn{i}" for i in range(1, NUM_CLIENTS + 1)]

# FedAvg에 기여하는 노드만 '학습 노드'다.
# 로컬 전용 노드는 학습은 하지만 집계 대기 대상이 아니다.
NON_CONTRIBUTING_NODES = list(
    dict.fromkeys(COLD_START_NODES + LOCAL_ONLY_NODES)
)

TRAINING_NODES = [
    n for n in ALL_NODES if n not in NON_CONTRIBUTING_NODES
]


# ════════════════════════════════════════════════════════
# 센서 데이터 출처
# ════════════════════════════════════════════════════════

# True면 MN-CSE의 contentInstance에서 실제 신호를 읽는다.
# False면 예전처럼 con에 실린 로컬 pkl 경로(data_path)를 읽는다.
# 경로 방식은 로봇/시뮬레이터와 학습 앱이 같은 머신에 있을 때만 동작한다.
FL_SENSOR_FROM_CSE = os.getenv(
    "FL_SENSOR_FROM_CSE",
    "1",
) not in ("0", "false", "False")


# ════════════════════════════════════════════════════════
# oneM2M 가중치 전송 설정
# ════════════════════════════════════════════════════════

# True면 모델 가중치를 contentInstance의 con 안에 base64로 실어 보낸다.
# False면 기존처럼 로컬 파일 경로만 전달한다(같은 PC에서만 동작).
FL_EMBED_WEIGHTS = os.getenv(
    "FL_EMBED_WEIGHTS",
    "1",
) not in ("0", "false", "False")

# 가중치를 fp16으로 줄여 보낼지 여부. 전송량이 절반이 된다.
# 정확도가 민감하면 0으로 끄면 된다.
FL_WEIGHTS_FP16 = os.getenv(
    "FL_WEIGHTS_FP16",
    "1",
) not in ("0", "false", "False")

# contentInstance 하나에 담을 base64 가중치의 상한(바이트).
# 이 값을 넘으면 embed를 포기하고 경로 전달로 되돌아간다.
#
# ACME 기준으로 넉넉히 둔다. CVAE1D 가중치가 base64로 약 217KB이므로 통과한다.
# 이 값을 CSE의 실제 페이로드 한계보다 크게 잡으면 CIN 생성이 실패하고
# 라운드 전체가 멈춘다.
FL_MAX_CIN_BYTES = int(
    os.getenv(
        "FL_MAX_CIN_BYTES",
        str(4 * 1024 * 1024),
    )
)

# 가중치가 위 상한을 넘을 때 조각내어 보낼지 여부.
#
# 기본값은 꺼짐(0)이다. ACME에는 작은 페이로드 상한이 없고 CVAE1D 가중치가
# base64로 약 217KB라 contentInstance 하나에 그대로 들어간다.
# 페이로드 상한이 작은 CSE로 옮길 때만 켠다(조각마다 전용 컨테이너를 두고
# 각각의 la를 읽는 방식으로 검증해 둠: 39조각 업로드 0.8s / 회수 0.1s).
FL_CHUNK_WEIGHTS = os.getenv(
    "FL_CHUNK_WEIGHTS",
    "0",
) not in ("0", "false", "False")


# ════════════════════════════════════════════════════════
# 설정 인스턴스
# ════════════════════════════════════════════════════════

TRAIN_CFG = TrainConfig()
AE_CFG = AEConfig()