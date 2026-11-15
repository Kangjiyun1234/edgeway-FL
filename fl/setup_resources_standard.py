"""
fl/setup_resources_standard.py — oneM2M FL 리소스 구조 생성 (ACME, MEP별 CSE)

MEC 샌드박스에는 CSE가 4개다. 사업장마다 MEP 안에 MN-CSE가 하나씩 있고,
클라우드에 IN-CSE가 하나 있다. 리소스마다 어느 CSE에 올라가야 하는지가
다르므로, 이 스크립트는 생성 직전에 바인딩을 바꿔 가며 돈다.

  IN-CSE (클라우드)                        config.in_endpoint()
    IN-AE                                  ← 우리가 만든다
      cnt-fl-control      라운드 지시 (IN→MN)
      cnt-global-model    글로벌 가중치 (IN→MN)
      cnt-local-updates   로컬 업데이트 수집 (MN→IN)
        cnt-mn1/2/3       노드별 dropbox
      cnt-fl-inference    판정 결과 (대시보드·샌드박스가 읽는다)

  MN-CSE (mep-a/b/c)                       config.mn_endpoint(node)
    siteA_AMR_1 … siteA_AMR_4              ← Isaac Sim이 만든다. 확인만 한다
      telemetry                            ← Isaac이 채운다. 우리는 읽는다
      status
    MN-AE-1                                ← 우리가 만든다
      cnt-local-model     라운드 캐시
      cnt-net-status      MEC028 스냅샷 / 버퍼 상태

AE가 두 종류라는 점이 중요하다.
  로봇 AE   : Isaac Sim이 CSE에 등록한다. 여기서 만들지 않는다. 없으면 경고만.
  FL 앱 AE  : MN-AE-1/2/3, IN-AE. 아무도 만들어 주지 않으므로 여기서 만든다.

TinyIoT 시절의 PostgreSQL 직접 초기화(clean_db)는 없앴다. ACME는 클라우드에
있고 DB 접근권도 없다. 초기화는 oneM2M DELETE로만 한다.

ACP 정책
  acp-fl-control    IN-AE 전체 / MN-AE-* RETRIEVE          (IN-CSE)
  acp-global-model  IN-AE 전체 / MN-AE-* RETRIEVE          (IN-CSE)
  acp-local-updates IN-AE 전체 / MN-AE-* CREATE            (IN-CSE)
  acp-dropbox-mnX   IN-AE 전체 / 해당 MN-AE-X만 CREATE     (IN-CSE)
  acp-cache-mnX     해당 MN-AE-X 전체 / IN-AE RETRIEVE     (MN-CSE of X)
  로봇 telemetry에는 ACP를 걸지 않는다. Isaac 소유다.

ACOP 비트마스크 (TS-0001)
  CREATE 1 / RETRIEVE 2 / UPDATE 4 / DELETE 8 / NOTIFY 16 / DISCOVERY 32 / ALL 63

사용법
  python3 fl/setup_resources_standard.py              전체 생성
  python3 fl/setup_resources_standard.py --verify      확인만
  python3 fl/setup_resources_standard.py --clean       우리 리소스 삭제 후 재생성
  python3 fl/setup_resources_standard.py --nodes mn1,mn2
"""

import argparse
import sys
import time
from typing import Dict, List, Optional

import config
import onem2m_utils as om2m


ORIGIN_IN_AE = "CIN-AE"

ACOP_CREATE = 1
ACOP_RETRIEVE = 2
ACOP_UPDATE = 4
ACOP_DELETE = 8
ACOP_NOTIFY = 16
ACOP_DISCOVERY = 32
ACOP_ALL = 63

SETTLE = 0.3  # CSE가 생성을 커밋할 시간


def origin_mn(node: str) -> str:
    """MN-AE-i의 Originator. mn2 -> CMN-AE-2"""
    return f"CMN-AE-{node[2:]}"


def all_mn_origins(nodes: List[str]) -> List[str]:
    return [origin_mn(n) for n in nodes]


# ════════════════════════════════════════════════════════
# CSE 바인딩
# ════════════════════════════════════════════════════════

def bind_in() -> str:
    """이후의 요청을 IN-CSE(클라우드)로 보낸다. CSE 이름을 반환."""
    base, cse = config.in_endpoint()
    om2m.set_endpoint(base, cse)
    om2m.set_in_endpoint(base, cse)
    return cse


def bind_mn(node: str) -> str:
    """이후의 요청을 해당 노드의 MEP MN-CSE로 보낸다. CSE 이름을 반환."""
    base, cse = config.mn_endpoint(node)
    om2m.set_endpoint(base, cse)
    in_base, in_cse = config.in_endpoint()
    om2m.set_in_endpoint(in_base, in_cse)
    return cse


def reachable(label: str) -> bool:
    """현재 바인딩된 CSE의 베이스를 RETRIEVE 해서 살아 있는지 본다."""
    cse = om2m.cse_name()
    code, body = om2m.probe_resource(cse)
    if code != 200:
        print(f"  ✗ {label}: {om2m.base_url()}/{cse}")
        if code is None:
            print(f"      연결 실패 — {body}")
        else:
            print(f"      HTTP {code}  {body}")
            hint = {
                400: "헤더가 잘못됐을 수 있다 (RETRIEVE에 Content-Type ty)",
                401: "인증 필요 — FL_IN_USER / FL_IN_PASS 를 설정하라",
                403: "권한 없음 — X-M2M-Origin 과 ACP를 확인하라",
                404: "경로에 리소스가 없다 — CSE 이름을 확인하라",
            }.get(code)
            if hint:
                print(f"      → {hint}")
        return False
    print(f"  ✓ {label}: {om2m.base_url()}/{cse}")
    return True


# ════════════════════════════════════════════════════════
# AE 생성 (우리 소유: IN-AE, MN-AE-i)
# ════════════════════════════════════════════════════════

def create_fl_ae(ae_name: str, origin: str) -> bool:
    """
    FL 애플리케이션 AE를 등록한다.

    rr=True는 request reachable. IN-AE가 보내는 NOTIFY를 MN-AE가 받으려면
    필요하다. srv는 지원 릴리스 버전(TS-0004).
    이미 있으면(409) 그대로 둔다.
    """
    cse = om2m.cse_name()
    payload = {
        "m2m:ae": {
            "rn": ae_name,
            "api": f"N{ae_name}",
            "rr": True,
            "srv": ["2a", "3"],
        }
    }
    headers = om2m._headers(2)
    headers["X-M2M-Origin"] = origin

    try:
        # _request_with_retry 를 거쳐야 TLS 설정과 기본 인증이 같이 적용된다.
        r = om2m._request_with_retry(
            "post",
            om2m._url(cse),
            json=payload,
            headers=headers,
            timeout=10,
        )
    except Exception as e:
        print(f"  ✗ AE {ae_name}: {e}")
        return False

    if r.status_code == 201:
        print(f"  ✓ AE 생성: {ae_name}  (Origin {origin})")
        return True
    if r.status_code == 409:
        print(f"  = AE 이미 있음: {ae_name}")
        return True
    print(f"  ✗ AE {ae_name}: {r.status_code} {r.text[:160]}")
    return False


# ════════════════════════════════════════════════════════
# 로봇 AE 확인 (Isaac Sim 소유 — 만들지 않는다)
# ════════════════════════════════════════════════════════

def verify_robot_aes(node: str) -> Dict[str, bool]:
    """
    현재 바인딩된 MN-CSE에서 이 사업장의 로봇 AE와 telemetry 컨테이너를 찾는다.

    Isaac Sim이 등록하므로 여기서는 디스커버리(fu=1&ty=2)로 목록만 보고,
    기대 대수와 비교해 보고한다. 없으면 Isaac을 먼저 돌리라고 알린다.
    """
    cse = om2m.cse_name()
    site = config.SITE_OF_NODE.get(node, node)
    want = config.ROBOTS_PER_NODE.get(node, 0)

    uril = om2m.discover(cse, ty=2, lim=200)
    robots = sorted({
        p.rstrip("/").split("/")[-1]
        for p in uril
        if p.rstrip("/").split("/")[-1].startswith(f"{site}_")
    })

    print(f"  로봇 AE ({site}_*): {len(robots)}/{want}대 발견")

    found: Dict[str, bool] = {}
    for rb in robots:
        tel = f"{cse}/{rb}/{config.ROBOT_TELEMETRY_CNT}"
        ok = om2m.resource_exists(tel)
        found[rb] = ok
        print(f"    {'✓' if ok else '✗'} {rb}/{config.ROBOT_TELEMETRY_CNT}")

    if not robots:
        print(f"    ⚠ {site} 로봇 AE가 하나도 없다. Isaac Sim을 먼저 띄워라.")
    elif len(robots) < want:
        print(f"    ⚠ {want}대 중 {len(robots)}대만 등록됐다.")

    return found


# ════════════════════════════════════════════════════════
# IN-CSE 쪽 ACP
# ════════════════════════════════════════════════════════

def setup_in_acps(nodes: List[str]) -> Dict[str, str]:
    """
    IN-CSE에 FL 제어용 ACP를 만든다. 반환은 {논리이름: ACP ri}.

    pv는 이 ACP가 붙은 리소스에 대한 접근 규칙, pvs는 ACP 자신에 대한
    규칙이다. pvs를 좁게 주지 않으면 누구든 정책을 고칠 수 있다.
    """
    print("\n=== IN-CSE ACP ===")
    cse = bind_in()
    admin = config.ORIGINATOR
    mn_origins = all_mn_origins(nodes)
    acps: Dict[str, str] = {}

    def mk(name: str, pv: List[Dict]) -> None:
        ri = om2m.create_acp(
            cse, name, pv,
            [{"acor": [admin, ORIGIN_IN_AE], "acop": ACOP_ALL}],
        )
        if ri:
            acps[name.replace("acp-", "")] = ri
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}")
        time.sleep(SETTLE)

    read_for_mn = [
        {"acor": [admin, ORIGIN_IN_AE], "acop": ACOP_ALL},
        {"acor": mn_origins,
         "acop": ACOP_RETRIEVE | ACOP_NOTIFY | ACOP_DISCOVERY},
    ]
    mk("acp-fl-control", read_for_mn)
    mk("acp-global-model", read_for_mn)

    mk("acp-local-updates", [
        {"acor": [admin, ORIGIN_IN_AE], "acop": ACOP_ALL},
        {"acor": mn_origins, "acop": ACOP_CREATE | ACOP_DISCOVERY},
    ])

    for node in nodes:
        mk(f"acp-dropbox-{node}", [
            {"acor": [admin, ORIGIN_IN_AE], "acop": ACOP_ALL},
            {"acor": [origin_mn(node)],
             "acop": ACOP_CREATE | ACOP_RETRIEVE},
        ])

    return acps


def setup_mn_acp(node: str) -> Optional[str]:
    """노드 자신의 MN-CSE에 로컬 캐시용 ACP를 만든다."""
    cse = om2m.cse_name()
    ri = om2m.create_acp(
        cse, f"acp-cache-{node}",
        [
            {"acor": [config.ORIGINATOR, origin_mn(node)], "acop": ACOP_ALL},
            {"acor": [ORIGIN_IN_AE],
             "acop": ACOP_RETRIEVE | ACOP_DISCOVERY},
        ],
        [{"acor": [config.ORIGINATOR, origin_mn(node)], "acop": ACOP_ALL}],
    )
    print(f"  {'✓' if ri else '✗'} acp-cache-{node}")
    time.sleep(SETTLE)
    return ri


# ════════════════════════════════════════════════════════
# IN-CSE 리소스
# ════════════════════════════════════════════════════════

def setup_in_resources(nodes: List[str], acps: Dict[str, str]) -> None:
    """IN-AE와 그 아래 제어·모델·수집 컨테이너를 만든다."""
    print("\n=== IN-CSE 리소스 ===")
    cse = bind_in()

    create_fl_ae(config.IN_AE_NAME, ORIGIN_IN_AE)
    time.sleep(SETTLE)

    ae = f"{cse}/{config.IN_AE_NAME}"

    def cnt(parent: str, name: str, mni: int, acp_key: Optional[str]) -> None:
        om2m.create_container(
            parent, name, mni=mni,
            acpi=[acps[acp_key]] if acp_key and acps.get(acp_key) else None,
        )
        print(f"  ✓ {name}  (mni={mni})")
        time.sleep(SETTLE)

    cnt(ae, "cnt-fl-control", 50, "fl-control")
    cnt(ae, "cnt-global-model", 50, "global-model")
    cnt(ae, "cnt-local-updates", 50, "local-updates")

    # 노드별 dropbox. 한 노드가 다른 노드의 업데이트를 보지 못하게 분리한다.
    updates = f"{ae}/cnt-local-updates"
    for node in nodes:
        om2m.create_container(
            updates, f"cnt-{node}", mni=50,
            acpi=([acps[f"dropbox-{node}"]]
                  if acps.get(f"dropbox-{node}") else None),
        )
        print(f"    ✓ cnt-local-updates/cnt-{node}")
        time.sleep(SETTLE)

    # 판정 결과. FL은 명령을 내리지 않고 여기에 써 두기만 한다.
    # 샌드박스/Isaac이 읽어서 2차 로봇 투입·대기장소 이동을 수행한다.
    cnt(ae, "cnt-fl-inference", 200, None)


# ════════════════════════════════════════════════════════
# MN-CSE 리소스
# ════════════════════════════════════════════════════════

def setup_mn_resources(node: str) -> Dict[str, bool]:
    """
    한 사업장 분. MEP의 MN-CSE에 바인딩하고
      1) 로봇 AE / telemetry 확인 (Isaac 소유)
      2) MN-AE-i 생성 (우리 소유)
      3) cnt-local-model, cnt-net-status 생성
    """
    site = config.SITE_OF_NODE.get(node, node)
    mep = config.MEP_OF_NODE.get(node, "?")
    print(f"\n=== {node} / {site} / {mep} ===")

    cse = bind_mn(node)
    if not reachable(f"{mep} MN-CSE"):
        return {}

    robots = verify_robot_aes(node)

    ae_name = f"MN-AE-{node[2:]}"
    create_fl_ae(ae_name, origin_mn(node))
    time.sleep(SETTLE)

    acp_ri = setup_mn_acp(node)
    ae = f"{cse}/{ae_name}"

    # 라운드별 로컬 모델 캐시. mni=5면 최근 5라운드만 남는다.
    om2m.create_container(
        ae, "cnt-local-model", mni=5,
        acpi=[acp_ri] if acp_ri else None,
    )
    print("  ✓ cnt-local-model  (mni=5)")
    time.sleep(SETTLE)

    # MEC028 스냅샷과 버퍼 상태. 시나리오 2·3이 여기에 기록한다.
    om2m.create_container(
        ae, "cnt-net-status", mni=100,
        acpi=[acp_ri] if acp_ri else None,
    )
    print("  ✓ cnt-net-status  (mni=100)")
    time.sleep(SETTLE)

    return robots


# ════════════════════════════════════════════════════════
# 확인
# ════════════════════════════════════════════════════════

def verify(nodes: List[str]) -> bool:
    print("\n" + "=" * 60)
    print("  리소스 확인")
    print("=" * 60)

    ok = True

    cse = bind_in()
    print(f"\n[IN-CSE] {om2m.base_url()}/{cse}")
    ae = f"{cse}/{config.IN_AE_NAME}"
    targets = [
        ae,
        f"{ae}/cnt-fl-control",
        f"{ae}/cnt-global-model",
        f"{ae}/cnt-local-updates",
        f"{ae}/cnt-fl-inference",
    ] + [f"{ae}/cnt-local-updates/cnt-{n}" for n in nodes]

    for t in targets:
        hit = om2m.resource_exists(t)
        ok &= hit
        print(f"  {'✓' if hit else '✗'} {t.split('/', 1)[-1]}")

    for node in nodes:
        cse = bind_mn(node)
        site = config.SITE_OF_NODE.get(node, node)
        mep = config.MEP_OF_NODE.get(node, "?")
        print(f"\n[{mep}] {om2m.base_url()}/{cse}  ({site})")

        ae_name = f"MN-AE-{node[2:]}"
        for t in (f"{cse}/{ae_name}",
                  f"{cse}/{ae_name}/cnt-local-model",
                  f"{cse}/{ae_name}/cnt-net-status"):
            hit = om2m.resource_exists(t)
            ok &= hit
            print(f"  {'✓' if hit else '✗'} {t.split('/', 1)[-1]}")

        # 로봇은 Isaac 소유라 없어도 실패로 치지 않는다. 경고만.
        robots = verify_robot_aes(node)
        if not robots:
            print("    (로봇 데이터 없이는 학습을 돌릴 수 없다)")

    print()
    print("✓ 우리 리소스는 모두 있다" if ok else "✗ 빠진 리소스가 있다")
    return ok


# ════════════════════════════════════════════════════════
# 초기화 (우리 리소스만)
# ════════════════════════════════════════════════════════

def clean(nodes: List[str]) -> None:
    """
    oneM2M DELETE로 우리가 만든 것만 지운다.
    로봇 AE는 Isaac 소유이므로 건드리지 않는다.
    """
    print("\n=== 우리 리소스 삭제 ===")

    cse = bind_in()
    for name in (f"{cse}/{config.IN_AE_NAME}",):
        print(f"  {'✓' if om2m.delete_resource(name) else '✗'} {name}")
    for node in nodes:
        for n in (f"acp-dropbox-{node}",):
            om2m.delete_resource(f"{cse}/{n}")
    for n in ("acp-fl-control", "acp-global-model", "acp-local-updates"):
        om2m.delete_resource(f"{cse}/{n}")
    print("  ✓ IN-CSE ACP 삭제")

    for node in nodes:
        cse = bind_mn(node)
        ae_name = f"MN-AE-{node[2:]}"
        print(f"  {'✓' if om2m.delete_resource(f'{cse}/{ae_name}') else '✗'} "
              f"{config.MEP_OF_NODE.get(node)}/{ae_name}")
        om2m.delete_resource(f"{cse}/acp-cache-{node}")


# ════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true",
                    help="우리 리소스를 지우고 다시 만든다")
    ap.add_argument("--verify", action="store_true",
                    help="생성하지 않고 확인만 한다")
    ap.add_argument("--nodes", default=",".join(config.ALL_NODES),
                    help="대상 노드 (기본: mn1,mn2,mn3)")
    args = ap.parse_args()

    config.require_sandbox()

    nodes = [n.strip() for n in args.nodes.split(",") if n.strip()]

    print("=" * 60)
    print("  oneM2M FL 리소스 설정 (ACME / MEP별 CSE)")
    print("=" * 60)
    print(f"  샌드박스 : {config.SANDBOX_HOST} / {config.SANDBOX_NAME}")
    print(f"  노드     : {', '.join(nodes)}")
    print(f"  학습 참여: {', '.join(config.TRAINING_NODES)}")
    print(f"  비참여   : {', '.join(config.NON_CONTRIBUTING_NODES)}")

    # 먼저 4개 CSE가 다 뜨는지 본다. 하나라도 못 뜨면 생성이 반쯤
    # 끝난 상태로 남아 더 헷갈린다.
    print("\n=== CSE 연결 확인 ===")
    bind_in()
    alive = reachable("IN-CSE")
    for node in nodes:
        bind_mn(node)
        alive &= reachable(f"{config.MEP_OF_NODE.get(node)} MN-CSE")

    if not alive:
        print("\n✗ 닿지 않는 CSE가 있다. 샌드박스 시나리오가 떠 있는지,")
        print("  FL_SANDBOX_HOST / FL_SANDBOX_NAME 이 맞는지 확인하라.")
        sys.exit(1)

    if args.verify:
        sys.exit(0 if verify(nodes) else 1)

    if args.clean:
        clean(nodes)
        time.sleep(1.0)

    acps = setup_in_acps(nodes)
    setup_in_resources(nodes, acps)
    for node in nodes:
        setup_mn_resources(node)

    ok = verify(nodes)
    print("\n다음: python3 fl/in_ae_standard.py  /  python3 fl/mn_ae_standard.py <node>")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
