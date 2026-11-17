"""
fl/valset.py — 고정 검증셋으로 글로벌 모델 성능을 잰다.

[왜 고정셋인가]
지금까지 성능 수치(val_loss, val_auroc)는 그 라운드에 받은 윈도우로 계산했다.
라운드마다 데이터가 바뀌니 R3 의 AUROC 와 R9 의 AUROC 를 비교할 수 없었다.
라벨도 라이브 스트림에는 안 붙어 있어 늘 단일 클래스 -> AUROC 0.5 였다.

여기서는 라벨이 붙은 고정 윈도우 묶음(fl/valset_amr.npz)을 매 라운드 같은
자로 재서, "라운드가 갈수록 좋아지는가"를 수치로 남긴다.

[라벨을 어디에 쓰나]
채점에만 쓴다. 모델은 추론 때 라벨을 보지 않고, 정규화 통계와 임계값도
라벨 없이 검증셋 전체에서 뽑는다. 그래서 전 구간이 비지도다.

[정규화]
글로벌 모델에는 정규화 통계가 없다(사업장마다 따로 갖는다). 그래서 검증셋
자체의 채널별 평균/표준편차로 표준화한다.

학습 때 쓴 자와 다르다. 전에 여기 "AUROC 는 순위 기반이라 영향받지 않는다"
고 적혀 있었는데 틀렸다. 순위가 매겨지는 대상은 점수이고, 정규화는 점수를
내기 전에 입력을 바꾸므로 순위 자체가 바뀐다.

실측(한 노드, 40에폭, 시드 4개): 평가도 학습과 같은 통계로 정규화하면
AUROC 0.7146 ± 0.0197, 검증셋 자기 통계로 하면 0.6852 ± 0.0166. 약 0.03
손해이고 시드 4개 모두 같은 방향이었다.

고치려면 글로벌 모델과 함께 정규화 통계를 실어 보내야 하는데, 사업장마다
통계가 다르므로 "어느 사업장의 자로 잴 것인가"를 먼저 정해야 한다. 지금은
그대로 두고 이 차이를 알고 읽는다. 절대 오차값을 사업장 임계값과 직접
비교하지는 말 것.

[파일 형식]
  signals (N, 38, 60) float32   윈도우
  cond    (N, 8)      float32   조건 벡터
  labels  (N,)        int64     0 정상 / 1 이상
  (선택) robot, scn, cond_name, run_id
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import config
from model_cvae import CVAE1D

DEFAULT_PATH = str(Path(__file__).resolve().parent / "valset_amr.npz")
VALSET_PATH = os.getenv("FL_VALSET_PATH", DEFAULT_PATH)

_cache = None


def load(path: Optional[str] = None):
    """검증셋을 한 번만 읽어 캐시한다. 없으면 None."""
    global _cache
    if _cache is not None:
        return _cache

    p = path or VALSET_PATH
    if not os.path.exists(p):
        return None

    try:
        z = np.load(p, allow_pickle=False)
        X = z["signals"].astype(np.float32)
        C = z["cond"].astype(np.float32)
        Y = z["labels"].astype(np.int64)
    except Exception as e:
        print(f"  ⚠ 검증셋 로드 실패 ({p}): {e}")
        return None

    if len(X) == 0 or len(np.unique(Y)) < 2:
        print(f"  ⚠ 검증셋이 비었거나 한 클래스뿐이다 ({p})")
        return None

    # 검증셋 자체 통계로 표준화. 라벨은 쓰지 않는다.
    mean = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2), keepdims=True) + 1e-8
    Xn = (X - mean) / std

    _cache = {
        "X": torch.from_numpy(Xn),
        "C": torch.from_numpy(C),
        "y": Y,
        "path": p,
    }
    print(f"  ✓ 검증셋 {len(Y)}개 "
          f"(정상 {int((Y == 0).sum())} / 이상 {int((Y == 1).sum())}) "
          f"<- {os.path.basename(p)}")
    return _cache


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.5
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores))
    except Exception:
        # sklearn 이 없으면 순위로 직접 센다 (Mann-Whitney U).
        order = np.argsort(scores, kind="stable")
        ranks = np.empty(len(scores), dtype=np.float64)
        ranks[order] = np.arange(1, len(scores) + 1)
        pos, neg = labels == 1, labels == 0
        n1, n0 = int(pos.sum()), int(neg.sum())
        if n1 == 0 or n0 == 0:
            return 0.5
        return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def evaluate(model_path: str, n_sigma: float = None) -> Optional[dict]:
    """
    글로벌 모델 하나를 검증셋으로 잰다.

    반환: {auroc, threshold, detected, false_alarms, miss,
           mean_normal, mean_anomaly, separation, n_normal, n_anomaly}
    """
    ds = load()
    if ds is None:
        return None

    if not model_path or not os.path.exists(model_path):
        return None

    cfg = config.AE_CFG
    n_sigma = (config.THRESHOLD_N_SIGMA if n_sigma is None else n_sigma)

    try:
        model = CVAE1D(
            n_channels=cfg.n_channels,
            cond_dim=cfg.cond_dim,
            latent_dim=cfg.latent_dim,
            seq_len=cfg.seq_len,
        )
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
    except Exception as e:
        print(f"  ⚠ 검증용 모델 로드 실패: {e}")
        return None

    X, C, y = ds["X"], ds["C"], ds["y"]

    with torch.no_grad():
        E = model.channel_error(X, C if cfg.cond_dim else None)
        # 채널별 규모가 달라 그냥 평균내면 큰 채널이 점수를 지배한다.
        # 기준은 검증셋 전체의 채널 평균이다. 라벨을 쓰지 않는다.
        E = E / (E.mean(dim=0) + 1e-12)
        scores = E.mean(dim=1).cpu().numpy()

    # 임계값도 라벨 없이 전체에서 뽑는다. 다만 평균/표준편차가 아니라
    # 중앙값/MAD 를 쓴다.
    #
    # 검증셋에는 정상과 이상이 섞여 있다. 평균과 표준편차는 이상치에 끌려
    # 올라가므로 임계값이 과하게 높아진다. 실제로 그 때문에 오경보는 0 인데
    # 탐지가 8% 에 그쳤다. 중앙값과 MAD 는 이상치가 섞여도 거의 움직이지
    # 않아 정상 분포를 그대로 반영한다.
    #
    # 1.4826 은 정규분포에서 MAD 를 표준편차와 같은 자로 맞추는 상수다.
    # 그래서 n_sigma 의 의미(3σ)가 그대로 유지된다.
    med = float(np.median(scores))
    mad = float(np.median(np.abs(scores - med))) * 1.4826
    if mad <= 0:
        mad = float(scores.std())      # 점수가 거의 같으면 MAD 가 0 이 된다
    thr = med + n_sigma * mad

    # 예전 기준도 같이 남긴다. 둘을 비교해야 임계값 때문인지 모델 때문인지
    # 가릴 수 있다.
    thr_mean = float(scores.mean() + n_sigma * scores.std())

    norm, anom = scores[y == 0], scores[y == 1]
    return {
        "auroc": round(_auroc(scores, y), 4),
        "threshold": round(thr, 4),
        "threshold_mean3s": round(thr_mean, 4),
        "detected_mean3s": int((anom > thr_mean).sum()),
        "false_alarms_mean3s": int((norm > thr_mean).sum()),
        "detected": int((anom > thr).sum()),
        "miss": int((anom <= thr).sum()),
        "false_alarms": int((norm > thr).sum()),
        "mean_normal": round(float(norm.mean()), 4),
        "mean_anomaly": round(float(anom.mean()), 4),
        "separation": round(float(anom.mean() - norm.mean()), 4),
        "n_normal": int(len(norm)),
        "n_anomaly": int(len(anom)),
    }


def summary_line(m: dict) -> str:
    """로그 한 줄."""
    return (f"AUROC {m['auroc']:.4f}  "
            f"탐지 {m['detected']}/{m['n_anomaly']}  "
            f"오경보 {m['false_alarms']}/{m['n_normal']}  "
            f"정상 {m['mean_normal']:.3f} vs 이상 {m['mean_anomaly']:.3f}  "
            f"thr {m['threshold']:.3f}"
            f"(평균3σ {m['threshold_mean3s']:.3f} -> "
            f"{m['detected_mean3s']}/{m['false_alarms_mean3s']})")
