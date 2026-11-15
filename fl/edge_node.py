"""
fl/edge_node.py — AMR 주행 데이터용 Edge 노드

AEEdgeNode: 조건부 1D CNN VAE 기반 정상 주행 재구성 학습 (이상 탐지)
  - 입력: 다채널 시계열 (N, 38, 60) + 조건 one-hot (N, 11)
  - 손실: ELBO (재구성 MSE + beta * KL), 정상 데이터만
  - 점수: 채널별 재구성 오차를 기준 오차로 정규화해 집계
  - FedAvg 집계: get/set_state_dict()

레거시:
  EdgeNode: 통계 피처 기반 MLP 분류기 (이전 버전, 미사용)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import TrainConfig, AEConfig
from model_cvae import CVAE1D

StateDict = dict[str, torch.Tensor]


# ════════════════════════════════════════════════════════
# AE Edge Node
# ════════════════════════════════════════════════════════

class AEEdgeNode:
    """
    조건부 1D CNN VAE 기반 Edge 노드

    학습: 정상 주행 데이터만 사용, ELBO(재구성 + KL) 최소화
    평가: 재구성 오차 기반 이상 점수로 AUROC / 임계값 산출

    입력 형태
        train_signals (N, C, T)  C=센서 채널(38), T=윈도우 길이(60)
        train_cond    (N, D)     조건 one-hot. 없으면 0 벡터로 채운다.
        (N, T) 2차원으로 들어오면 단일 채널로 보고 (N, 1, T)로 올린다.
    """

    def __init__(
        self,
        node_id:    int,
        train_signals: np.ndarray,
        val_signals:   np.ndarray,
        val_labels:    np.ndarray,
        ae_cfg:     AEConfig,
        train_cfg:  TrainConfig,
        device:     str = "cpu",
        train_cond: Optional[np.ndarray] = None,
        val_cond:   Optional[np.ndarray] = None,
        train_run_id: Optional[np.ndarray] = None,
        val_run_id:   Optional[np.ndarray] = None,
    ):
        self.node_id   = node_id
        self.device    = device
        self.ae_cfg    = ae_cfg
        self.train_cfg = train_cfg

        self.cond_dim = int(getattr(ae_cfg, "cond_dim", 0) or 0)
        self.beta     = float(getattr(ae_cfg, "beta", 0.5))
        self.topk     = int(getattr(ae_cfg, "score_topk", 0) or 0)

        X_train = self._as_3d(train_signals)
        X_val   = self._as_3d(val_signals)

        C_train = self._as_cond(train_cond, len(X_train))
        C_val   = self._as_cond(val_cond,   len(X_val))

        y_val = torch.tensor(np.asarray(val_labels).reshape(-1), dtype=torch.long)

        self.train_ds = TensorDataset(X_train, C_train)
        self.val_ds   = TensorDataset(X_val, C_val, y_val)

        self.train_run_id = (np.asarray(train_run_id)
                             if train_run_id is not None else None)
        self.val_run_id = (np.asarray(val_run_id)
                           if val_run_id is not None else None)

        self.num_train_samples = len(X_train)
        n_ch = X_train.shape[1] if len(X_train) else int(ae_cfg.n_channels)
        seq  = X_train.shape[2] if len(X_train) else int(ae_cfg.seq_len)

        self.model = CVAE1D(
            n_channels = n_ch,
            cond_dim   = self.cond_dim,
            latent_dim = ae_cfg.latent_dim,
            hidden     = int(getattr(ae_cfg, "hidden", 48)),
            seq_len    = seq,
        ).to(device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr           = train_cfg.lr,
            weight_decay = train_cfg.weight_decay,
        )

        # 채널별 기준 오차. 학습 후 정상 데이터에서 채운다.
        # 채널마다 오차 규모가 달라서, 정규화 없이 평균내면
        # 규모가 큰 채널이 점수를 지배한다.
        self.channel_norm: Optional[torch.Tensor] = None
        # 정상 데이터로 잡은 임계값
        self.threshold: Optional[float] = None

        print(f"    [AEEdgeNode] node={node_id}  train={self.num_train_samples}"
              f"  val={len(X_val)}  ch={n_ch}  seq={seq}"
              f"  cond={self.cond_dim}  latent={ae_cfg.latent_dim}")

    # ── 입력 정리 ─────────────────────────────────────────

    @staticmethod
    def _as_3d(a: np.ndarray) -> torch.Tensor:
        arr = np.asarray(a, dtype=np.float32)
        if arr.ndim == 2:          # (N, T) → 단일 채널
            arr = arr[:, None, :]
        elif arr.ndim == 1:
            arr = arr.reshape(0, 1, 0)
        return torch.tensor(arr, dtype=torch.float32)

    def _as_cond(self, c: Optional[np.ndarray], n: int) -> torch.Tensor:
        if self.cond_dim <= 0:
            return torch.zeros(n, 0, dtype=torch.float32)
        if c is None or len(c) != n:
            return torch.zeros(n, self.cond_dim, dtype=torch.float32)
        return torch.tensor(np.asarray(c, dtype=np.float32), dtype=torch.float32)

    # ── 가중치 공유 ───────────────────────────────────────

    def get_state_dict(self) -> StateDict:
        return {k: v.detach().cpu().clone()
                for k, v in self.model.state_dict().items()}

    def set_state_dict(self, state: StateDict) -> None:
        self.model.load_state_dict(state)
        self.model.to(self.device)

    # ── 로컬 학습 ─────────────────────────────────────────

    def train_local(
        self,
        dp_epsilon:       Optional[float] = None,
        dp_delta:         float = 1e-5,
        dp_max_grad_norm: float = 1.5,
    ) -> tuple[float, float, float]:
        """
        로컬 학습 (정상 데이터만). 반환: (train_loss, val_loss, val_auroc)
        """
        if self.num_train_samples == 0:
            print("    ⚠ 학습 데이터가 없다")
            v_loss, v_auroc = self._evaluate_val()
            return 0.0, float(v_loss), float(v_auroc)

        use_dp = dp_epsilon is not None
        if use_dp:
            sigma = math.sqrt(2 * math.log(1.25 / dp_delta)) / dp_epsilon

        train_loader = DataLoader(
            self.train_ds,
            batch_size = self.train_cfg.batch_size,
            shuffle    = True,
            # BatchNorm은 배치 크기 1에서 실패한다. 마지막 조각이 1이면 버린다.
            drop_last  = (self.num_train_samples % self.train_cfg.batch_size == 1),
        )

        self.model.train()
        loss_sum, total = 0.0, 0

        for _ in range(self.train_cfg.local_epochs):
            for xb, cb in train_loader:
                xb = xb.to(self.device)
                cb = cb.to(self.device) if self.cond_dim else None

                self.optimizer.zero_grad()
                loss, rec, kl = self.model.loss(xb, cb, beta=self.beta)

                bs = int(xb.size(0))
                loss_sum += float(loss.item()) * bs
                total    += bs

                loss.backward()

                if use_dp:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=dp_max_grad_norm
                    )
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad += (torch.randn_like(param.grad)
                                           * sigma * dp_max_grad_norm / bs)

                self.optimizer.step()

        train_loss = loss_sum / max(total, 1)

        # 학습이 끝난 모델 기준으로 채널 기준 오차와 임계값을 다시 잡는다.
        self.calibrate()

        val_loss, val_auroc = self._evaluate_val()
        return float(train_loss), float(val_loss), float(val_auroc)

    # ── 캘리브레이션 ──────────────────────────────────────

    def calibrate(self, n_sigma: float = 3.0) -> Optional[float]:
        """
        학습 정상 데이터로 채널별 기준 오차와 임계값을 정한다.

        임계값은 노드마다 자기 데이터로 잡아야 한다. 사업장·로봇마다
        정상의 모습이 달라서 공용 임계값을 쓰면 정상도 이상으로 나온다.
        """
        if self.num_train_samples == 0:
            return None

        self.model.eval()
        errs = []
        with torch.no_grad():
            for i in range(0, self.num_train_samples, self.train_cfg.batch_size):
                xb = self.train_ds.tensors[0][i:i + self.train_cfg.batch_size].to(self.device)
                cb = (self.train_ds.tensors[1][i:i + self.train_cfg.batch_size].to(self.device)
                      if self.cond_dim else None)
                errs.append(self.model.channel_error(xb, cb).cpu())

        E = torch.cat(errs)                       # (N, C)
        self.channel_norm = E.mean(dim=0)         # 채널별 기준 오차

        scores = self._aggregate(E)
        self.threshold = float(scores.mean() + n_sigma * scores.std())
        return self.threshold

    def _aggregate(self, E: torch.Tensor) -> torch.Tensor:
        """채널별 오차 (N, C) -> 이상 점수 (N,)"""
        if self.channel_norm is not None:
            E = E / (self.channel_norm.to(E.device) + 1e-12)
        if self.topk and 0 < self.topk < E.shape[1]:
            E = E.topk(self.topk, dim=1).values
        return E.mean(dim=1)

    # ── 평가 ──────────────────────────────────────────────

    def _evaluate_val(self) -> tuple[float, float]:
        """val 세트 평가: (정상 평균 점수, AUROC)"""
        if len(self.val_ds) == 0:
            return 0.0, 0.5

        val_loader = DataLoader(
            self.val_ds,
            batch_size = self.train_cfg.batch_size,
            shuffle    = False,
        )

        self.model.eval()
        all_scores, all_labels = [], []

        with torch.no_grad():
            for xb, cb, yb in val_loader:
                xb = xb.to(self.device)
                cb_ = cb.to(self.device) if self.cond_dim else None
                all_scores.append(self._aggregate(
                    self.model.channel_error(xb, cb_)).cpu())
                all_labels.append(yb)

        scores = torch.cat(all_scores).numpy()
        labels = torch.cat(all_labels).numpy()

        normal = scores[labels == 0]
        val_loss = float(normal.mean()) if len(normal) else 0.0

        return val_loss, _compute_auroc(scores, labels)

    def compute_anomaly_scores(
        self,
        signals: np.ndarray,
        cond: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """(N, C, T) 신호 -> 이상 점수 (N,)"""
        X = self._as_3d(signals).to(self.device)
        if len(X) == 0:
            return np.zeros((0,), dtype=np.float32)
        C = self._as_cond(cond, len(X)).to(self.device)

        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), self.train_cfg.batch_size):
                xb = X[i:i + self.train_cfg.batch_size]
                cb = C[i:i + self.train_cfg.batch_size] if self.cond_dim else None
                out.append(self._aggregate(
                    self.model.channel_error(xb, cb)).cpu().numpy())

        return np.concatenate(out)

    @staticmethod
    def aggregate_by_run(
        scores: np.ndarray,
        run_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        윈도우 점수를 주행 1회 단위로 평균낸다.

        실제 판단 단위는 1초가 아니라 "이 로봇이 이 구간을 지나는 동안"이다.
        검증 실험에서 윈도우 단위 AUROC 0.65가 주행 단위로는 0.73까지 올랐다.
        반환: (run별 점수, run id)
        """
        runs = np.asarray(run_ids)
        uniq = np.unique(runs)
        return (np.array([scores[runs == r].mean() for r in uniq]), uniq)


# ════════════════════════════════════════════════════════
# AUROC 헬퍼
# ════════════════════════════════════════════════════════

def _compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    scores가 높을수록 anomaly에 가깝다고 가정
    labels: 0=정상, 1=이상
    """
    if len(np.unique(labels)) < 2:
        return 0.5   # 단일 클래스면 의미 없음

    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores))
    except Exception:
        return 0.5


# ════════════════════════════════════════════════════════
# 레거시 EdgeNode (통계 피처 기반 MLP, 미사용)
# ════════════════════════════════════════════════════════

class EdgeNode:
    """레거시: 통계 피처 벡터 → 분류 (현재 미사용, 인터페이스 유지)"""

    def __init__(self, node_id, data, sensor_cols, train_cfg, device="cpu", **kwargs):
        raise NotImplementedError(
            "EdgeNode (통계피처 MLP) 는 더 이상 사용하지 않습니다. "
            "AEEdgeNode 를 사용하세요."
        )
