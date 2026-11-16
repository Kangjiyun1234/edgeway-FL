"""
fl/verdict.py — 이상 판정 결과를 사업장 CSE에 기록한다.

[어디에 쓰나]
판정 결과는 사업장 밖으로 나가지 않는다. 설계서가 사업장 내부에 보관할
것으로 "바닥 이상 위치, 로봇 점검 상태와 운행 지시"를 들고 있다.
그래서 중앙 IN-CSE 가 아니라 그 사업장의 MN-CSE 에 남긴다.

  <MN-CSE>/ANALYSIS-AE-1/cnt-verdict/<cin>

[왜 MN-AE 가 아니라 별도 AE 인가]
설계서에서 분석 앱과 학습 앱은 서로 다른 앱이고, ACP 도 역할별로 나뉘어야
한다. 지금은 모델을 들고 있는 MN-AE 가 판정까지 계산하지만, 쓸 때는 분석 앱
신원(CANALYSIS-AE-i)으로 쓴다. 나중에 분석 앱을 별도 프로세스로 떼어내도
리소스 트리와 구독하는 쪽 코드는 그대로다.

[언제 올리나 — 윈도우마다가 아니다]
재구성 오차가 임계값을 '연속 K회' 넘은 그 지점에서 한 번만 올린다
(config.ANOMALY_K_CONSECUTIVE, 기본 3). 한 번 정상으로 떨어졌다가 다시 K회를
채우면 그때 또 올린다.

이렇게 하지 않으면 이상이 이어지는 동안 윈도우 수만큼 CIN 이 쌓인다. 공장당
500~600 윈도우 규모에서 구독자가 같은 사실을 수백 번 받게 된다.

[구독하는 쪽]
cnt-verdict 에 <subscription>(net=3)을 걸면 새 판정마다 알림을 받는다.
알림 본문의 m2m:sgn/nev/rep/m2m:cin/con 이 아래 payload 의 JSON 문자열이다.
lbl 에 로봇 이름과 판정이 들어가므로 특정 로봇 것만 디스커버리할 수도 있다.

  <cnt-verdict>?fu=1&ty=4&lbl=siteA_AMR_1
"""

import os
import time
from typing import List, Optional, Sequence

import config
import onem2m_utils as om2m


# 연속 몇 회를 넘겨야 판정으로 올릴지. 오프라인 분석과 같은 값을 쓴다.
K_CONSECUTIVE = int(os.getenv(
    "FL_VERDICT_K", str(getattr(config, "ANOMALY_K_CONSECUTIVE", 3))))


def onset_indices(scores: Sequence[float], threshold: float,
                  k: Optional[int] = None) -> List[int]:
    """
    연속 k회 초과가 '시작된' 지점들의 인덱스.

    이어지는 동안은 다시 잡지 않는다. 끊겼다가 다시 k회를 채우면 그때 또 잡는다.
    k번째 윈도우에서 발화하므로 인덱스는 그 k번째를 가리킨다.
    """
    k = K_CONSECUTIVE if k is None else k
    hits: List[int] = []
    run = 0
    for i, s in enumerate(scores):
        if s > threshold:
            run += 1
            if run == k:
                hits.append(i)
        else:
            run = 0
    return hits


def publish(node_name: str,
            cnt_path: str,
            origin: str,
            scores,
            robots,
            threshold: float,
            round_num: int,
            cts=None,
            model_uri: Optional[str] = None,
            k: Optional[int] = None) -> int:
    """
    로봇별로 판정을 내고 CIN 으로 올린다. 올린 건수를 돌려준다.

    scores/robots/cts 는 같은 길이의 윈도우 단위 배열이다. 로봇마다 따로
    끊어서 보기 때문에, 한 로봇의 이상이 다른 로봇의 연속 판정에 섞이지 않는다.
    """
    if threshold is None or not len(scores):
        return 0

    k = K_CONSECUTIVE if k is None else k
    site = getattr(config, "SITE_OF_NODE", {}).get(node_name, node_name)
    total = 0

    for robot in sorted({str(r) for r in robots}):
        idx = [i for i, r in enumerate(robots) if str(r) == robot]
        if not idx:
            continue
        s_robot = [float(scores[i]) for i in idx]

        for hit in onset_indices(s_robot, threshold, k):
            i = idx[hit]
            score = float(scores[i])
            payload = {
                "type": "amr-verdict",
                "site": site,
                "node": node_name,
                "robot": robot,
                "verdict": "ANOMALY",
                "score": score,
                "threshold": float(threshold),
                "ratio": round(score / threshold, 3) if threshold else None,
                "consecutive": k,
                "window_index": i,
                "observed_at": (str(cts[i]) if cts is not None
                                and i < len(cts) else None),
                "round": round_num,
                "model": model_uri,
                "timestamp": time.time(),
            }
            res = om2m.create_content_instance(
                cnt_path, payload,
                labels=[robot, "ANOMALY", site],
                origin=origin,
            )
            if res:
                total += 1
                print(f"    ⚠ 판정 {robot}: score={score:.6f} "
                      f"(임계 {threshold:.6f}, {k}회 연속)")

    if total:
        print(f"  ✓ 판정 {total}건 -> {om2m.strip_prefix(cnt_path)}")
    return total
