"""
prepare_data_amr.py — AMR 주행 데이터를 FL 노드(사업장)별 PKL로 변환

입력 형식 두 가지를 모두 받는다.

  1) CIN 덤프 (새 형식, 권장)
     <robot>/telemetry.json  — JSON Lines. 한 줄이 contentInstance 하나.
       {"ri":…, "rn":…, "ct":…, "lbl":null, "con":{…}}
     디렉터리를 주면 그 아래 */telemetry.json 을 모두 읽는다.

  2) merged.json (옛 형식)
     run 목록이고 run 하나가 CIN 여러 개를 갖는다. lbl에 scn/spd/load가 있다.

출력: /tmp/fl_data/amr/{mn1,mn2,mn3}.pkl

[라벨에 대하여]
운영 중에는 cond:normal 같은 라벨이 붙지 않는다. 로봇은 그냥 주행하고
데이터를 올릴 뿐이고, 이상 탐지는 정상 주행만 학습하는 비지도 방식이다.
그래서 이 스크립트도 라벨을 요구하지 않는다.

평가용 정답(over_bump 등)은 학습에 쓰지 않고 AUROC를 재는 데만 쓴다.
시뮬레이터가 아는 결함 위치를 결과 확인용으로만 쓰는 것과 같다.

[정지 구간]
nav 라벨이 아니라 실제 주행 속도로 가려낸다. 그래야 주행 중 잠깐 멈춘
구간은 살아남고 진짜 정지만 빠진다. 기준은 AMR_MIN_SPEED.

[중단된 주행]
nav=aborted 는 중단된 주행이라 정상 주행으로 보기 어렵다. 학습에서는 빼고
평가에는 남긴다. AMR_EXCLUDE_NAV 로 조정한다.

[사업장 분할]
로봇 식별자로 나눈다. A 4대 / B 2대 / C 4대.
siteA_* / siteB_* / siteC_* 규칙을 따르거나 AMR_ROBOT_MAP으로 지정한다.
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import sys

import numpy as np

import amr_schema as sch

# 하위 호환: 기존 코드가 prepare_data_amr.CH_SPEC 등을 참조한다.
CH_SPEC = sch.CH_SPEC
CH_NAMES = sch.CH_NAMES
IS_POS = sch.IS_POS
N_CH = sch.N_CH
T = sch.T
COND_DIM = sch.COND_DIM
cond_vector = sch.cond_vector

SRC = os.getenv(
    "AMR_DATA",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "amr_data"),
)
OUT_DIR = os.getenv("AMR_OUT_DIR", "/tmp/fl_data/amr")

NORMAL = "normal"
ANOMALY_NAV = {"over_bump"}          # 평가 정답으로 쓸 nav

NODES = ["mn1", "mn2", "mn3"]
SITE_OF_NODE = {"mn1": "siteA", "mn2": "siteB", "mn3": "siteC"}
NODE_OF_SITE = {v: k for k, v in SITE_OF_NODE.items()}
EXPECTED_ROBOTS = {"mn1": 4, "mn2": 2, "mn3": 4}

VAL_RATIO = float(os.getenv("AMR_VAL_RATIO", "0.2"))
SEED = int(os.getenv("AMR_SEED", "42"))

_fallback_warned = False


def _parse_robot_map(spec: str) -> dict:
    out = {}
    for part in spec.split(";"):
        if "=" not in part:
            continue
        node, robots = part.split("=", 1)
        for r in robots.split(","):
            if r.strip():
                out[r.strip()] = node.strip()
    return out


ROBOT_MAP = _parse_robot_map(os.getenv("AMR_ROBOT_MAP", ""))


def site_of_robot(robot: str) -> str:
    global _fallback_warned
    if robot in ROBOT_MAP:
        return ROBOT_MAP[robot]
    prefix = robot.split("_", 1)[0] if "_" in robot else ""
    if prefix in NODE_OF_SITE:
        return NODE_OF_SITE[prefix]
    if not _fallback_warned:
        print(f"  ⚠ 로봇 식별자로 사업장을 정할 수 없다 (예: {robot!r}) -> mn1로 보낸다.")
        print("    siteA_* / siteB_* / siteC_* 규칙을 쓰거나 AMR_ROBOT_MAP을 지정하라.")
        _fallback_warned = True
    return "mn1"


# ── 입력 읽기 ─────────────────────────────────────────────
def _iter_cin_dump(path: str):
    """JSON Lines CIN 덤프에서 (con, lbl)을 흘려준다."""
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            con = rec.get("con")
            if isinstance(con, str):
                try:
                    con = json.loads(con)
                except ValueError:
                    continue
            if isinstance(con, dict):
                yield con, (rec.get("lbl") or [])


def _iter_merged(path: str):
    """옛 merged.json에서 (con, lbl)을 흘려준다. run 라벨을 CIN에 얹는다."""
    with open(path, encoding="utf-8") as fp:
        data = json.load(fp)
    runs = data if isinstance(data, list) else data.get("runs", [])
    for run in runs:
        run_lbl = list(run.get("lbl", []))
        for cin in run.get("cins", []):
            con = cin.get("con")
            if isinstance(con, str):
                try:
                    con = json.loads(con)
                except ValueError:
                    continue
            if isinstance(con, dict):
                yield con, (cin.get("lbl") or run_lbl)


def iter_records(src: str):
    """파일/디렉터리 어느 쪽이든 (con, lbl)을 흘려준다."""
    if os.path.isdir(src):
        files = sorted(glob.glob(os.path.join(src, "**", "telemetry.json"),
                                 recursive=True))
        if files:
            for f in files:
                yield from _iter_cin_dump(f)
            return
        merged = os.path.join(src, "merged.json")
        if os.path.exists(merged):
            yield from _iter_merged(merged)
            return
        raise FileNotFoundError(f"{src} 아래에서 telemetry.json 도 merged.json 도 못 찾았다")

    name = os.path.basename(src).lower()
    if name.startswith("merged"):
        yield from _iter_merged(src)
    else:
        yield from _iter_cin_dump(src)


# ── 파싱 ──────────────────────────────────────────────────
def parse(src: str) -> dict:
    X, C, NAV, ROBOT, POSE, NODE, LABEL = [], [], [], [], [], [], []
    skipped = 0

    for con, lbl in iter_records(src):
        sig = sch.window_from_con(con)
        if sig is None:
            skipped += 1
            continue

        labels = {x.split(":")[0]: x.split(":", 1)[1]
                  for x in lbl if isinstance(x, str) and ":" in x}

        nav = con.get("nav") or labels.get("scn") or "idle"
        speed = sch.speed_of(sig)

        # 조건 벡터: 라벨이 있으면 쓰고, 없으면 payload에서 끌어낸다.
        if "scn" in labels and "spd" in labels:
            cvec = sch.cond_from_labels(labels["scn"], labels["spd"])
        else:
            cvec = sch.cond_vector(nav, speed)

        # 평가용 정답. 라벨이 있으면 라벨 우선, 없으면 nav로 판단.
        cond_name = labels.get(
            "cond",
            "floor_bump" if nav in ANOMALY_NAV else NORMAL,
        )

        X.append(sig)
        C.append(cvec)
        NAV.append(nav)
        ROBOT.append(con.get("robot", "unknown"))
        POSE.append(sch.pose_of(con))
        NODE.append(site_of_robot(con.get("robot", "unknown")))
        LABEL.append(cond_name)

    return {
        "X": np.asarray(X, dtype=np.float32),
        "C": np.asarray(C, dtype=np.float32),
        "nav": np.asarray(NAV),
        "robot": np.asarray(ROBOT),
        "pose": np.asarray(POSE, dtype=np.float32),
        "node": np.asarray(NODE),
        "label": np.asarray(LABEL),
        "skipped": skipped,
    }


# ── 노드별 PKL ────────────────────────────────────────────
def build_node_pkl(d: dict, node: str, rng: np.random.Generator) -> dict:
    sel = d["node"] == node
    if not sel.any():
        return {}

    X, C = d["X"][sel], d["C"][sel]
    nav, robot = d["nav"][sel], d["robot"][sel]
    pose, label = d["pose"][sel], d["label"][sel]

    # 학습 후보: 움직이는 구간 + 이상 아님 + 제외 nav 아님
    moving = sch.moving_mask(X)
    is_normal = label == NORMAL
    not_excluded = ~np.isin(nav, list(sch.EXCLUDE_NAV_TRAIN))
    trainable = moving & is_normal & not_excluded

    idx = np.where(trainable)[0]
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * VAL_RATIO)) if len(idx) else 0
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    tr = np.zeros(len(X), bool); tr[tr_idx] = True
    va = np.zeros(len(X), bool); va[val_idx] = True
    # 평가: 움직이는 구간 전부 (검증용 정상 + 이상 + 중단 포함)
    te = moving & (va | ~is_normal | ~not_excluded)

    if tr.any():
        mean = X[tr].mean(axis=(0, 2), keepdims=True)
        std = X[tr].std(axis=(0, 2), keepdims=True)
    else:
        mean = np.zeros((1, N_CH, 1), np.float32)
        std = np.ones((1, N_CH, 1), np.float32)
    std = std + 1e-8

    def norm(a):
        return ((a - mean) / std).astype(np.float32)

    return {
        "train_signals": norm(X[tr]), "train_cond": C[tr],
        "train_nav": nav[tr], "train_robot": robot[tr],

        "val_signals": norm(X[va]), "val_cond": C[va],
        "val_labels": np.zeros(int(va.sum()), dtype=np.int64),
        "val_nav": nav[va],

        "test_signals": norm(X[te]), "test_cond": C[te],
        "test_labels": (label[te] != NORMAL).astype(np.int64),
        "test_label_name": label[te], "test_nav": nav[te],
        "test_pose": pose[te], "test_robot": robot[te],

        "norm_mean": mean.astype(np.float32),
        "norm_std": std.astype(np.float32),
        "channel_names": CH_NAMES, "cond_dim": COND_DIM,
        "n_channels": N_CH, "seq_len": T,
        "node": node, "site": SITE_OF_NODE.get(node, node),
        "robots": sorted(set(robot.tolist())),
        "spd_edges": list(sch.SPD_EDGES), "min_speed": sch.MIN_SPEED,
    }


def main() -> int:
    if not os.path.exists(SRC):
        print(f"✗ 데이터가 없다: {SRC}")
        print("  AMR_DATA 로 telemetry.json 또는 그 상위 디렉터리를 지정하라.")
        return 1

    print(f"입력: {SRC}")
    d = parse(SRC)
    if not len(d["X"]):
        print("✗ 유효한 윈도우가 없다")
        return 1

    moving = sch.moving_mask(d["X"])
    print(f"  윈도우 {d['X'].shape}  (건너뜀 {d['skipped']}개)")
    print(f"  이동 {int(moving.sum())} / 정지 {int((~moving).sum())}"
          f"  (기준 속도 {sch.MIN_SPEED})")
    u, c = np.unique(d["label"], return_counts=True)
    print("  상태별:", dict(zip(u.tolist(), c.tolist())))
    u, c = np.unique(d["robot"], return_counts=True)
    print("  로봇별:", dict(zip(u.tolist(), c.tolist())))

    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    for node in NODES:
        pkl = build_node_pkl(d, node, rng)
        if not pkl:
            print(f"  ⚠ {node}({SITE_OF_NODE[node]}): 배정된 데이터가 없다")
            continue
        want = EXPECTED_ROBOTS.get(node)
        n_rb = len(pkl["robots"])
        warn = "" if (want is None or n_rb == want) else f" ⚠예상{want}대"
        path = os.path.join(OUT_DIR, f"{node}.pkl")
        with open(path, "wb") as fp:
            pickle.dump(pkl, fp)
        print(f"  ✓ {node}({pkl['site']}) 로봇{n_rb}대{warn}: "
              f"학습 {len(pkl['train_signals'])} / 검증 {len(pkl['val_signals'])} / "
              f"평가 {len(pkl['test_signals'])} (이상 {int(pkl['test_labels'].sum())})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
