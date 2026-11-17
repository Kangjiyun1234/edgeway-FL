# fl/aggregator.py
"""
FedAvg 집계.

[가중 방식]
표준 FedAvg는 클라이언트의 샘플 수에 비례해 가중한다. 중앙에 데이터를 모아
학습한 결과를 근사하는 것이 목적이기 때문이다.

이 데모는 그렇게 하지 않는다. 참여 주체가 '사업장'이고, 사업장마다 로봇
대수가 다르다(A 4대, B 2대, C 4대). 로봇이 많으면 같은 시간에 더 많은
윈도우가 쌓이므로, 샘플 수로 가중하면 로봇을 많이 둔 사업장이 공통 모델을
더 끌어당긴다. 대수는 그 공장의 사정일 뿐 발언권의 근거가 아니다.

그래서 사업장마다 같은 무게를 준다. 기관 단위로 연합하는 경우(cross-silo FL)
에서 흔히 쓰는 방식이다. 데이터를 버리지 않으면서 기여를 맞출 수 있다.

표준 방식으로 되돌리려면 FL_AGG_WEIGHTING=samples 로 실행한다.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import torch

StateDict = Dict[str, torch.Tensor]

# "uniform" 사업장마다 동일 가중 (기본)
# "samples" 샘플 수 비례 — 표준 FedAvg
WEIGHTING = os.getenv("FL_AGG_WEIGHTING", "uniform").strip().lower()

# BatchNorm 층을 평균에서 뺄지 (FedBN).
#
# [왜]
# BatchNorm 은 배치의 평균/분산으로 정규화하고 그 통계를 running_mean /
# running_var 에 쌓는다. 그건 그 사업장 데이터의 통계다. 사업장마다 바닥도
# 로봇도 다르므로 그 통계도 다르다. 평균을 내면 어느 사업장의 것도 아닌
# 값이 된다.
#
# 이 코드베이스는 입력 정규화를 이미 사업장별로 한다. 공유로 바꿔 재 봤더니
# 2노드 FedAvg 에서 AUROC 0.6752 -> 0.5772 로 떨어졌고 50 회 비교 전부
# 사업장별이 이겼다. BatchNorm 통계도 같은 성격인데 그것만 평균내고 있었다.
#
# 실측(2노드 FedAvg, 시드 4개): 평균을 내면 10R평균 0.6745 / R10 0.6850,
# 빼면 0.6815 / 0.7080. 오경보는 41 -> 36. 차이는 작지만 방향이 일관되고,
# base 는 R3 에서 평평해지는 반면 FedBN 은 R10 까지 계속 오른다.
#
# FL_FEDBN=0 으로 예전 동작(전부 평균)으로 되돌린다.
FEDBN = os.getenv("FL_FEDBN", "1") not in ("0", "false", "False")


def bn_prefixes(state) -> tuple:
    """
    state_dict 에서 BatchNorm 층의 접두사를 찾는다.

    running_mean 은 BatchNorm 에만 있으므로 그 키로 층을 특정한다.
    모델 클래스를 몰라도 되고, 층 이름이 바뀌어도 따라간다.
    """
    return tuple(k[: -len("running_mean")]
                 for k in state if k.endswith("running_mean"))


class Aggregator:
    """FedAvg Aggregator (모델 비의존적)"""

    def __init__(self, weighting: str | None = None):
        self.weighting = (weighting or WEIGHTING).lower()
        if self.weighting not in ("uniform", "samples"):
            raise ValueError(
                f"FL_AGG_WEIGHTING 은 uniform 또는 samples 여야 한다: "
                f"{self.weighting}"
            )

    def _weights(self, counts: List[int]) -> List[float]:
        """
        집계 가중치를 정한다. 합이 1이 되도록 정규화한다.

        uniform: 참여한 노드 수로 균등 분배. 샘플 수는 보지 않는다.
                 다만 샘플이 0인 노드는 학습을 못 한 것이므로 뺀다.
        samples: 샘플 수에 비례. 표준 FedAvg.
        """
        if self.weighting == "samples":
            total = sum(counts)
            if total == 0:
                raise ValueError("Total samples is zero")
            return [c / total for c in counts]

        active = [1.0 if c > 0 else 0.0 for c in counts]
        total = sum(active)
        if total == 0:
            raise ValueError("학습한 노드가 하나도 없다")
        return [a / total for a in active]

    def aggregate(
        self,
        client_states: List[Tuple[StateDict, int]],
    ) -> StateDict:
        """
        가중 평균으로 집계한다.

        client_states: [(state_dict, num_samples), ...]
        반환: 집계된 global state_dict
        """
        if not client_states:
            raise ValueError("No client states to aggregate")

        states = [state for state, _ in client_states]
        counts = [int(n) for _, n in client_states]
        weights = self._weights(counts)

        global_state: StateDict = {}
        bn = bn_prefixes(states[0]) if FEDBN else ()
        for key in states[0].keys():
            ref = states[0][key]
            if bn and key.startswith(bn):
                # BatchNorm 은 사업장에 남긴다. 여기 담는 값은 글로벌 모델을
                # 평가할 때만 쓰이고, 각 사업장은 자기 것을 계속 쓴다.
                global_state[key] = ref.clone()
                continue
            if not torch.is_floating_point(ref):
                # 정수 버퍼(num_batches_tracked 등)는 평균을 내면 뜻이 깨진다.
                # 첫 노드 것을 그대로 쓴다.
                global_state[key] = ref.clone()
                continue
            global_state[key] = sum(
                state[key] * w for state, w in zip(states, weights)
            )

        return global_state

    def describe(self, client_states: List[Tuple[StateDict, int]]) -> str:
        """집계에 쓴 가중치를 한 줄로 설명한다. 로그와 대시보드에 쓴다."""
        counts = [int(n) for _, n in client_states]
        try:
            weights = self._weights(counts)
        except ValueError as e:
            return f"집계 불가: {e}"
        parts = [f"n={c} w={w:.3f}" for c, w in zip(counts, weights)]
        label = ("사업장 균등" if self.weighting == "uniform"
                 else "샘플 수 비례")
        return f"{label}  " + "  ".join(parts)
