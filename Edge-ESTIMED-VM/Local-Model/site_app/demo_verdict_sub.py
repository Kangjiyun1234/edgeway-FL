"""
Edge-ESTIMED-VM/Local-Model/site_app/demo_verdict_sub.py — 판정 알림 경로를 샌드박스에서 한 번에 세워 보는 도구

무엇을 하나
  1) 구독자 AE 를 하나 만들고 <pollingChannel> 을 붙인다.
     (캠퍼스 방화벽 때문에 CSE 가 우리를 직접 못 부른다. 규격에 있는 길이다.)
  2) 진짜 ANALYSIS-AE-i/cnt-verdict 에 구독이 걸리는지 '재 본다'.
     - 로봇 AE-ID 로 한 번   : ACP 가 의도한 경로
     - 새로 만든 AE-ID 로 한 번: ACP 에 없는 신원. 막혀야 정상
     둘 다 만들었으면 바로 지운다. 남기지 않는다.
  3) 진짜 리소스를 건드리지 않는 시험용 트리를 따로 세운다.
     AE -> ACP -> container -> <subscription> -> <contentInstance>
     그리고 폴링 채널로 알림이 실제로 오는지 확인한다.

왜 '재 본다' 고 쓰나
  <subscription> 생성에 CREATE 가 필요한지 RETRIEVE 로 되는지는 CSE 구현이
  정한다. 문서를 읽고 단정하는 대신 실제 ACME 에 물어보고 HTTP 코드를 적는다.

실행
  source ~/venv-edgeway/bin/activate
  source .env.local
  python3 Edge-ESTIMED-VM/Local-Model/site_app/demo_verdict_sub.py            # 기본 mn3 (사업장 C)
  python3 Edge-ESTIMED-VM/Local-Model/site_app/demo_verdict_sub.py mn1        # 다른 사업장
"""
import json
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import config
import onem2m_utils as om2m
import onem2m_pch as pch

SUB_AE = "VERDICT-SUB-AE"
SUB_AE_ID = "CVERDICT-SUB-AE"
TEST_AE = "VERDICT-TEST-AE"
TEST_AE_ID = "CVERDICT-TEST-AE"
TEST_CNT = "cnt-verdict-test"
TEST_ACP = "acp-verdict-test"

ACOP_CREATE, ACOP_RETRIEVE = 1, 2
ACOP_NOTIFY, ACOP_DISCOVERY = 16, 32
ACOP_ALL = 63
NET_CREATE_OF_DIRECT_CHILD = 3


def head(msg: str) -> None:
    print(f"\n{'=' * 60}\n{msg}\n{'=' * 60}")


def create_ae(cse: str, name: str, aeid: str) -> bool:
    """AE 를 원하는 AE-ID 로 등록한다. 이미 있으면 그대로 둔다."""
    h = om2m._headers(2)
    h["X-M2M-Origin"] = aeid
    r = om2m._request_with_retry(
        "post", om2m._url(cse),
        json={"m2m:ae": {"rn": name, "api": f"N{name}",
                         "rr": True, "srv": ["2a", "3"]}},
        headers=h, timeout=10,
    )
    ok = r.status_code in (201, 409)
    print(f"  {'✓' if ok else '✗'} AE {name} ({aeid})  HTTP {r.status_code}")
    return ok


def try_subscribe(parent: str, name: str, nu: str, origin: str) -> int:
    """
    구독을 만들어 보고 HTTP 코드만 돌려준다. 성공/실패를 판정하지 않는다.
    판정은 호출한 쪽이 '기대'와 비교해서 한다.
    """
    r = om2m._request_with_retry(
        "post", om2m._url(parent),
        json={"m2m:sub": {"rn": name, "nu": [nu],
                          "enc": {"net": [NET_CREATE_OF_DIRECT_CHILD]}}},
        headers=om2m._headers(23, origin=origin), timeout=10,
    )
    return r.status_code


def drop(path: str, origin: str) -> None:
    try:
        om2m._request_with_retry(
            "delete", om2m._url(path),
            headers=om2m._headers(origin=origin), timeout=10)
    except Exception:
        pass


def main() -> None:
    node = sys.argv[1] if len(sys.argv) > 1 else "mn3"
    site = config.SITE_OF_NODE.get(node, node)

    base, cse = config.mn_endpoint(node)
    om2m.set_endpoint(base, cse)
    in_base, in_cse = config.in_endpoint()
    om2m.set_in_endpoint(in_base, in_cse)

    code, _ = om2m.probe_resource(cse)
    if code != 200:
        print(f"✗ MN-CSE 에 못 붙었다 (HTTP {code}). .env.local 을 확인하라.")
        return
    print(f"MN-CSE: {om2m.strip_prefix(cse)}   node={node} site={site}")

    # ── 1. 구독자 AE + 폴링 채널 ────────────────────────
    head("1. 구독자 AE 와 <pollingChannel>")
    sub_path = f"{cse}/{SUB_AE}"
    if not create_ae(cse, SUB_AE, SUB_AE_ID):
        return
    aei = pch.ae_identity(sub_path) or SUB_AE_ID
    chan = pch.find_channel(sub_path, aei) or pch.create_channel(sub_path, aei)
    if not chan:
        print("  ✗ 폴링 채널을 못 만들었다. 알림은 못 받는다.")
        return
    print(f"  ✓ pollingChannel: {chan}   aei={aei}")
    pch.set_request_unreachable(sub_path, aei, True)
    pcu = f"{sub_path}/{chan}/pcu"

    # ── 2. 진짜 cnt-verdict 에 구독이 걸리는지 측정 ──────
    head("2. 진짜 cnt-verdict 에 구독이 걸리는가 (측정)")
    real = f"{cse}/{config.analysis_ae_name(node)}/{config.VERDICT_CNT}"
    print(f"  대상: {om2m.strip_prefix(real)}")

    robots = []
    for robot in config.robot_names(node):
        res = om2m.get_resource(f"{cse}/{robot}")
        rid = ((res or {}).get("m2m:ae") or {}).get("aei")
        if rid:
            robots.append((robot, rid))
    print(f"  로봇 AE-ID: {[r[1] for r in robots] or '없음 (Isaac 미기동)'}")

    probes = []
    if robots:
        probes.append(("로봇 (ACP 에 RETRIEVE|NOTIFY|DISCOVERY 있음)",
                       robots[0][1], "허용되길 기대"))
    probes.append(("시험용 AE (ACP 에 없는 신원)", SUB_AE_ID, "차단되길 기대"))

    for label, origin, expect in probes:
        name = f"sub-probe-{int(time.time() * 1000) % 100000}"
        sc = try_subscribe(real, name, pch.notification_target(SUB_AE_ID),
                           origin)
        verdict = "생성됨" if sc == 201 else ("이미 있음" if sc == 409
                                           else f"거부 {sc}")
        print(f"    {label}\n      Origin={origin}  ->  HTTP {sc} ({verdict})"
              f"   [{expect}]")
        if sc in (201, 409):
            drop(f"{real}/{name}", origin)
            print(f"      · 흔적을 지웠다 ({name})")

    # ── 3. 시험용 트리 ───────────────────────────────────
    head("3. 시험용 트리 (진짜 리소스는 안 건드린다)")
    if not create_ae(cse, TEST_AE, TEST_AE_ID):
        return

    acp_ri = om2m.create_acp(
        cse, TEST_ACP,
        [
            {"acor": [config.ORIGINATOR, TEST_AE_ID], "acop": ACOP_ALL},
            # 구독자에게 cnt-verdict 와 같은 조합을 준다.
            {"acor": [SUB_AE_ID],
             "acop": ACOP_RETRIEVE | ACOP_NOTIFY | ACOP_DISCOVERY},
        ],
        [{"acor": [config.ORIGINATOR, TEST_AE_ID], "acop": ACOP_ALL}],
    )
    print(f"  {'✓' if acp_ri else '✗'} ACP {TEST_ACP}  ri={acp_ri}")

    cnt = f"{cse}/{TEST_AE}/{TEST_CNT}"
    om2m.create_container(f"{cse}/{TEST_AE}", TEST_CNT, mni=10,
                          acpi=[acp_ri] if acp_ri else None)
    print(f"  ✓ container {TEST_CNT}  (mni=10)")

    # 구독은 두 신원으로 시도한다. 하나는 컨테이너 주인, 하나는 구독자.
    for label, origin in (("컨테이너 주인", TEST_AE_ID),
                          ("구독자 (RETRIEVE|NOTIFY|DISCOVERY)", SUB_AE_ID)):
        name = f"sub-{'owner' if origin == TEST_AE_ID else 'reader'}"
        drop(f"{cnt}/{name}", TEST_AE_ID)
        sc = try_subscribe(cnt, name, pch.notification_target(SUB_AE_ID),
                           origin)
        print(f"  {'✓' if sc in (201, 409) else '✗'} sub {name}"
              f"  Origin={origin}  HTTP {sc}")

    # ── 4. 진짜 판정과 같은 모양의 CIN ───────────────────
    head("4. CIN 하나 올리고 알림을 받아 본다")
    robot = robots[0][0] if robots else f"{site}_AMR_1"
    payload = {
        "type": "amr-verdict",
        "site": site, "node": node, "robot": robot,
        "verdict": "ANOMALY",
        "score": 33.416542053222656,
        "threshold": 12.01379680633545,
        "ratio": 2.782,
        "consecutive": 3,
        "window_index": 34,
        "observed_at": "20261117T154558,121880",
        "pose": {"x": 63.842, "y": 14.051, "basis": "window_mean"},
        "round": 2,
        "model": None,
        "timestamp": time.time(),
    }
    res = om2m.create_content_instance(
        cnt, payload, labels=[robot, "ANOMALY", site], origin=TEST_AE_ID)
    print(f"  {'✓' if res else '✗'} CIN -> {om2m.strip_prefix(cnt)}")
    print(f"    con = {json.dumps(payload, ensure_ascii=False)[:120]}…")

    print("\n  폴링 채널에서 알림을 기다린다 (최대 20초)…")
    got = False
    deadline = time.time() + 20
    while time.time() < deadline and not got:
        try:
            r = om2m._request_with_retry(
                "get", om2m._url(pcu),
                headers=om2m._headers(origin=SUB_AE_ID), timeout=8)
        except Exception as e:
            print(f"    ⚠ 폴링 오류: {e}")
            break
        if r.status_code == 200 and r.text.strip():
            body = r.json()
            sgn = (body.get("m2m:rqp", {}).get("pc", {})
                   or body).get("m2m:sgn", {})
            cin = (sgn.get("nev", {}).get("rep", {}) or {}).get("m2m:cin", {})
            print(f"    ✓ 알림 도착")
            print(f"      sur = {sgn.get('sur')}")
            print(f"      con = {str(cin.get('con'))[:160]}…")
            got = True
        time.sleep(1)
    if not got:
        print("    ⚠ 20초 안에 알림이 안 왔다. 구독이 걸렸는지 위를 보라.")

    head("정리")
    print(f"  남긴 것: {TEST_AE} / {SUB_AE} / {TEST_ACP}  (MN-CSE {node})")
    print("  지우려면 이 스크립트에 --clean 을 붙여 다시 실행하라.")


def clean(node: str) -> None:
    """시험용으로 만든 것만 지운다. 진짜 리소스는 손대지 않는다."""
    base, cse = config.mn_endpoint(node)
    om2m.set_endpoint(base, cse)
    for path, origin in ((f"{cse}/{TEST_AE}", TEST_AE_ID),
                         (f"{cse}/{SUB_AE}", SUB_AE_ID),
                         (f"{cse}/{TEST_ACP}", config.ORIGINATOR)):
        ok = om2m.delete_resource(path) if origin == config.ORIGINATOR else None
        if ok is None:
            om2m.set_originator(origin)
            ok = om2m.delete_resource(path)
            om2m.set_originator(None)
        print(f"  {'✓' if ok else '✗'} 삭제 {om2m.strip_prefix(path)}")


if __name__ == "__main__":
    if "--clean" in sys.argv:
        clean(next((a for a in sys.argv[1:] if not a.startswith("-")), "mn3"))
    else:
        main()
