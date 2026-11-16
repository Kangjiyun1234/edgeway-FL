#!/usr/bin/env python3
"""
fl/probe_cra.py — MN-AE 가 쓰는 조회를 그대로 재현해서 어디서 0 이 되는지 본다.

증상: check_telemetry 는 커서 이후 7건을 찾는데 MN-AE 는 0건이다.
      둘은 같은 discover_cins_after 를 쓰지만 lim 이 다르다 (300 vs 50).

ACME 가 lim 을 cra 보다 먼저 적용하면, 컨테이너에 오래된 CIN 이 많을 때
앞에서 lim 개를 자른 뒤 cra 로 거르므로 결과가 0 이 된다. 그걸 확인한다.

사용:
    cd ~/projects/edgeway-FL
    set -a; source .env.local; set +a
    python3 fl/probe_cra.py mn2
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config                      # noqa: E402
import onem2m_utils as om2m        # noqa: E402
import amr_ingest                  # noqa: E402

node = sys.argv[1] if len(sys.argv) > 1 else "mn2"

om2m.set_endpoint(*config.mn_endpoint(node))
cse = config.mn_endpoint(node)[1]
rb = config.robot_names(node)[0]
path = f"{cse}/{rb}/{config.ROBOT_TELEMETRY_CNT}"

print("=" * 64)
print(f"  {node} / {rb}")
print(f"  {path}")
print("=" * 64)

res = om2m.get_resource(path)
cnt = (res or {}).get("m2m:cnt") or {}
print(f"  cni={cnt.get('cni')}  mni={cnt.get('mni')}")

last = om2m.get_latest_content_instance(path)
cursor = ((last or {}).get("m2m:cin") or {}).get("ct")
print(f"  커서로 쓸 ct = {cursor}")

print("\n  10초 기다려 새 CIN 을 쌓는다 ...")
time.sleep(10)

print("\n--- ① 같은 커서, lim 만 바꿔서 조회 ---")
for lim in (10, 50, 100, 300, 1000):
    uril = om2m.discover_cins_after(path, cursor, lim=lim)
    print(f"  lim={lim:<5} -> {len(uril or [])}건")

print("\n--- ② 커서 없이 (1라운드와 같은 경로) ---")
for lim in (50, 300):
    uril = om2m.list_uril(path, lim=lim)
    print(f"  lim={lim:<5} -> {len(uril or [])}건")

print("\n--- ③ CIN 을 실제로 윈도우로 만들어 본다 (MN-AE 와 같은 경로) ---")
got = amr_ingest.fetch_new_windows(path, cursor, 50)
print(f"  fetch_new_windows(lim=50)  -> 윈도우 {len(got['signals'])}개  "
      f"max_ct={got.get('max_ct')}")
got = amr_ingest.fetch_new_windows(path, cursor, 300)
print(f"  fetch_new_windows(lim=300) -> 윈도우 {len(got['signals'])}개  "
      f"max_ct={got.get('max_ct')}")

print("\n--- 판정 ---")
a = len(om2m.discover_cins_after(path, cursor, lim=50) or [])
b = len(om2m.discover_cins_after(path, cursor, lim=300) or [])
if a == 0 and b > 0:
    print("  lim 이 작으면 0 이 된다. ACME 가 lim 을 cra 보다 먼저 적용한다.")
    print("  => MN-AE 의 per_robot(=50) 이 원인이다. lim 을 키워야 한다.")
elif a > 0 and b > 0:
    print("  lim 과 무관하게 조회는 된다. 원인은 조회가 아니라")
    print("  윈도우를 만드는 쪽이거나 커서를 넘기는 쪽이다. 위 ③ 을 보라.")
else:
    print("  둘 다 0 이다. cra 자체가 안 듣는다.")
