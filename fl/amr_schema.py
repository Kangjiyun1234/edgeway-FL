"""
amr_schema.py — AMR 주행 데이터의 채널 규격과 조건 벡터 정의

prepare_data_amr(오프라인 파일)과 amr_ingest(라이브 CSE)가 같은 정의를 쓰도록
한 곳에 모았다. 채널을 바꾸거나 조건 구성을 바꿀 때 여기만 고치면 된다.

[채널 38개]
    IMU   front/left/right × (ax ay az gx gy gz roll pitch) = 24
    ENC   drive(pos,vel,eff) swivel(pos) roller_fl/fr/rl/rr(pos,vel) lift(pos,eff) = 14

[조건 벡터 8차원]
    nav 그룹 5 : 정지 / 직진 / 회전 / 적재·하역 / 기타
    속도 버킷 3 : 저 / 중 / 고   (drive.vel 절대값 평균)

운영 중에는 cond:normal 같은 라벨이 붙지 않는다. 로봇은 그냥 주행하고
데이터를 올릴 뿐이다. 그래서 조건은 payload 안에서 끌어낸다. nav와 속도는
추론 시점에도 그대로 얻을 수 있다.

over_bump 는 조건에 넣지 않는다. 넣으면 "여기가 단차다"를 모델에 알려주는
셈이라 이상 탐지가 성립하지 않는다. 직진 그룹으로 묶는다.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Tuple

import numpy as np

# ── 채널 ──────────────────────────────────────────────────
CH_SPEC = [("imu", p, c) for p in ("front", "left", "right")
           for c in ("ax", "ay", "az", "gx", "gy", "gz", "roll", "pitch")]
CH_SPEC += [("enc", "drive", c) for c in ("pos", "vel", "eff")]
CH_SPEC += [("enc", "swivel", "pos")]
CH_SPEC += [("enc", f"roller_{s}", c)
            for s in ("fl", "fr", "rl", "rr") for c in ("pos", "vel")]
CH_SPEC += [("enc", "lift", c) for c in ("pos", "eff")]

CH_NAMES = [f"{p}.{c}" for _, p, c in CH_SPEC]
IS_POS = np.array([c == "pos" for _, _, c in CH_SPEC])
N_CH = len(CH_SPEC)
VEL_IDX = CH_NAMES.index("drive.vel")

T = int(os.getenv("AMR_WINDOW", "60"))

# ── 조건 ──────────────────────────────────────────────────
NAV_GROUPS = ["stop", "straight", "turn", "loadunload", "other"]
N_NAV = len(NAV_GROUPS)
N_SPD = 3
COND_DIM = N_NAV + N_SPD

# nav 문자열 -> 그룹. 모르는 값은 'other'.
NAV_TO_GROUP = {
    "idle": 0, "done": 0, "reset": 0, "stopped": 0,
    "to_pick": 1, "to_drop": 1, "to_aisle": 1, "to_lane": 1,
    "straight": 1,
    # 단차 통과는 '직진'으로 둔다. 별도 그룹으로 만들면 정답 유출이다.
    "over_bump": 1,
    "turn_in": 2, "turn_out": 2, "turn_drop": 2, "turn_unload": 2,
    "turn_start": 2, "turn_lane": 2, "left": 2, "right": 2,
    "loading": 3, "unloading": 3,
    "aborted": 4,
}

# 속도 버킷 경계. 학습과 추론이 같은 값을 써야 한다.
SPD_EDGES = tuple(
    float(x) for x in os.getenv("AMR_SPD_EDGES", "2.0,6.0").split(",")
)

# 이 속도 아래는 '정지'로 보고 뺀다.
# nav 라벨이 아니라 실제 속도로 판정하므로, 주행 중 잠깐 멈춘 구간은 살아남는다.
MIN_SPEED = float(os.getenv("AMR_MIN_SPEED", "0.1"))

# 학습에서 뺄 nav. 중단된 주행은 정상 주행으로 보기 어렵다.
EXCLUDE_NAV_TRAIN = {
    n.strip() for n in os.getenv("AMR_EXCLUDE_NAV", "aborted").split(",")
    if n.strip()
}


def nav_group(nav: Optional[str]) -> int:
    return NAV_TO_GROUP.get(str(nav or "").strip(), 4)


def speed_of(window: np.ndarray) -> float:
    """윈도우 하나의 평균 주행 속도 (drive.vel 절대값)."""
    return float(np.abs(window[VEL_IDX]).mean())


def speed_bucket(speed: float) -> int:
    if speed < SPD_EDGES[0]:
        return 0
    if speed < SPD_EDGES[1]:
        return 1
    return 2


def cond_vector(nav: Optional[str], speed: float) -> np.ndarray:
    v = np.zeros(COND_DIM, dtype=np.float32)
    v[nav_group(nav)] = 1.0
    v[N_NAV + speed_bucket(speed)] = 1.0
    return v


# ── 레거시 라벨 변환 ──────────────────────────────────────
# 옛 데이터(merged.json)는 scn/spd/load 라벨을 갖고 있다.
# 새 구성과 맞추기 위해 같은 8차원으로 옮긴다. load는 새 데이터에 없으므로 버린다.
LEGACY_SCN_TO_NAV = {
    "idle": "idle", "straight": "straight", "left": "left", "right": "right",
}
LEGACY_SPD_TO_SPEED = {"none": 0.0, "low": 1.0, "mid": 4.0, "high": 8.0}


def cond_from_labels(scn: str, spd: str) -> np.ndarray:
    return cond_vector(
        LEGACY_SCN_TO_NAV.get(scn, scn),
        LEGACY_SPD_TO_SPEED.get(spd, 0.0),
    )


# ── 윈도우 추출 ───────────────────────────────────────────
def window_from_con(payload: dict) -> Optional[np.ndarray]:
    """
    CIN의 con(dict) 하나 -> 신호 (C, T). 채널이 모자라면 None.
    누적 위치(.pos)는 윈도우 평균을 빼서 '이 구간의 변화'만 남긴다.
    """
    cols = []
    for grp, part, chan in CH_SPEC:
        seq = payload.get(grp, {}).get(part, {}).get(chan)
        if not isinstance(seq, list) or len(seq) < T:
            return None
        cols.append([0.0 if v is None else float(v) for v in seq[:T]])

    sig = np.asarray(cols, dtype=np.float32)
    sig[IS_POS, :] -= sig[IS_POS, :].mean(axis=1, keepdims=True)
    return sig


def pose_of(payload: dict) -> Tuple[float, float]:
    px = [v for v in (payload.get("pose", {}).get("x") or []) if v is not None]
    py = [v for v in (payload.get("pose", {}).get("y") or []) if v is not None]
    return (float(np.mean(px)) if px else float("nan"),
            float(np.mean(py)) if py else float("nan"))


def moving_mask(X: np.ndarray) -> np.ndarray:
    """실제 속도로 정지 구간을 가려낸다."""
    if len(X) == 0:
        return np.zeros(0, dtype=bool)
    return np.array([speed_of(w) >= MIN_SPEED for w in X])
