#!/usr/bin/env python3
"""
fl/probe_drop.py — CIN 이 윈도우로 안 만들어지고 버려지는지 본다.

window_from_con 은 CH_SPEC 의 채널 중 하나라도 샘플이 T(60)개 미만이면
CIN 전체를 None 으로 버린다. 그런데 아무 로그도 안 남긴다. 그래서
"윈도우 50개 / CIN 92개" 처럼 숫자가 안 맞아도 이유를 알 수 없었다.

어느 채널이 몇 개 모자라서 버려지는지 세어 준다.

사용:
    cd ~/projects/edgeway-FL
    set -a; source .env.local; set +a
    python3 fl/probe_drop.py mn1
"""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config                      # noqa: E402
import onem2m_utils as om2m        # noqa: E402
import amr_ingest as ing           # noqa: E402
import amr_schema as sch           # noqa: E402

node = sys.argv[1] if len(sys.argv) > 1 else "mn1"
LIM = int(os.getenv("PROBE_LIM", "200"))

om2m.set_endpoint(*config.mn_endpoint(node))
cse = config.mn_endpoint(node)[1]

print("=" * 70)
print(f"  {node} / {config.SITE_OF_NODE.get(node, node)}   (로봇당 최근 {LIM}개)")
print(f"  윈도우 길이 T = {sch.T},  채널 {len(sch.CH_SPEC)}개")
print("=" * 70)

site_ok = site_drop = 0

for rb in config.robot_names(node):
    path = f"{cse}/{rb}/{config.ROBOT_TELEMETRY_CNT}"
    cnt = (om2m.get_resource(path) or {}).get("m2m:cnt") or {}

    uril = om2m.list_uril(path, lim=LIM)
    ok = drop = 0
    short = Counter()      # 모자란 채널
    missing = Counter()    # 아예 없는 채널
    seqs = []

    for u in uril:
        res = om2m.get_resource(u.lstrip("/"))
        if not res or "m2m:cin" not in res:
            continue
        con = ing._con_of(res)
        if con is None:
            continue

        seqs.append(con.get("seq"))

        bad = False
        for grp, part, chan in sch.CH_SPEC:
            seq = con.get(grp, {}).get(part, {}).get(chan)
            key = f"{grp}.{part}.{chan}"
            if not isinstance(seq, list):
                missing[key] += 1
                bad = True
            elif len(seq) < sch.T:
                short[f"{key}({len(seq)})"] += 1
                bad = True
        if bad:
            drop += 1
        else:
            ok += 1

    site_ok += ok
    site_drop += drop
    total = ok + drop
    pct = (drop / total * 100) if total else 0

    sq = [s for s in seqs if isinstance(s, int)]
    print(f"\n  {rb}  cni={cnt.get('cni')}")
    print(f"    조회 {total}개 -> 윈도우 {ok}개, 버림 {drop}개 ({pct:.0f}%)")
    if sq:
        print(f"    seq 범위: {min(sq)} ~ {max(sq)}")
    for k, v in short.most_common(5):
        print(f"      짧음  {k}: {v}개")
    for k, v in missing.most_common(5):
        print(f"      없음  {k}: {v}개")

total = site_ok + site_drop
print("\n" + "=" * 70)
print(f"  사업장 합계: 조회 {total}개 -> 윈도우 {site_ok}개, "
      f"버림 {site_drop}개 ({(site_drop/total*100) if total else 0:.0f}%)")
if site_drop:
    print("  => 버려지는 CIN 이 있다. 위에 찍힌 채널이 원인이다.")
    print("     Isaac Sim 이 그 채널을 짧게 보내거나 안 보내고 있다.")
else:
    print("  => 버려지는 CIN 은 없다. 수집이 적은 건 다른 이유다.")
print("=" * 70)
