"""
fl/setup_resources_standard.py — oneM2M FL 리소스 구조 생성 (ACME, MEP별 CSE)

MEC 샌드박스에는 CSE가 4개다. 사업장마다 MEP 안에 MN-CSE가 하나씩 있고,
클라우드에 IN-CSE가 하나 있다. 리소스마다 어느 CSE에 올라가야 하는지가
다르므로, 이 스크립트는 생성 직전에 바인딩을 바꿔 가며 돈다.

  IN-CSE (클라우드)                        config.in_endpoint()
    TRAINING-AE-1/2/3                      ← 우리가 만든다. 사업장 학습 앱이
                                             MN-CSE 와 IN-CSE 양쪽에 등록된다.
                                             IN 쪽은 <pollingChannel> 을 걸고
                                             드롭박스에 쓰기 위한 신원이다.
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
    TRAINING-AE-1                          ← 우리가 만든다. 학습 앱
      cnt-local-model     라운드 캐시
      cnt-net-status      MEC028 스냅샷 / 버퍼 상태
    ANALYSIS-AE-1                          ← 우리가 만든다. 분석 앱
      cnt-verdict         판정 결과. 사업장 밖으로 나가지 않는다

AE가 두 종류라는 점이 중요하다.
  로봇 AE   : Isaac Sim이 CSE에 등록한다. 여기서 만들지 않는다. 없으면 경고만.
  FL 앱 AE  : TRAINING-AE-1/2/3, ANALYSIS-AE-1/2/3, IN-AE.
              아무도 만들어 주지 않으므로 여기서 만든다.

TinyIoT 시절의 PostgreSQL 직접 초기화(clean_db)는 없앴다. ACME는 클라우드에
있고 DB 접근권도 없다. 초기화는 oneM2M DELETE로만 한다.

ACP 정책
  acp-fl-control    IN-AE 전체 / TRAINING-AE-* RETRIEVE          (IN-CSE)
  acp-global-model  IN-AE 전체 / TRAINING-AE-* RETRIEVE          (IN-CSE)
  acp-local-updates IN-AE 전체 / TRAINING-AE-* CREATE            (IN-CSE)
  acp-dropbox-mnX   IN-AE 전체 / 해당 TRAINING-AE-X만 CREATE     (IN-CSE)
  acp-cache-mnX     해당 TRAINING-AE-X 전체 / IN-AE RETRIEVE     (MN-CSE of X)
  acp-verdict-mnX   분석 앱 전체 / 학습 앱·로봇 AE RETRIEVE      (MN-CSE of X)
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
import os
import sys
import time
from typing import Dict, List, Optional

import config
import onem2m_utils as om2m


ORIGIN_IN_AE = config.IN_AE_ID

ACOP_CREATE = 1
ACOP_RETRIEVE = 2
ACOP_UPDATE = 4
ACOP_DELETE = 8
ACOP_NOTIFY = 16
ACOP_DISCOVERY = 32
ACOP_ALL = 63

SETTLE = 0.3  # CSE가 생성을 커밋할 시간


def origin_mn(node: str) -> str:
    """학습 앱의 Originator. mn2 -> CTRAINING-AE-2"""
    return config.training_ae_id(node)


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
            print(f"      connection failed — {body}")
        else:
            print(f"      HTTP {code}  {body}")
            hint = {
                400: "header may be wrong (Content-Type ty on a RETRIEVE)",
                401: "authentication required — set FL_IN_USER / FL_IN_PASS",
                403: "forbidden — check X-M2M-Origin and the ACP",
                404: "no resource at this path — check the CSE name",
            }.get(code)
            if hint:
                print(f"      → {hint}")
        return False
    print(f"  ✓ {label}: {om2m.base_url()}/{cse}")
    return True


# ════════════════════════════════════════════════════════
# AE 생성 (우리 소유: IN-AE, TRAINING-AE-i, ANALYSIS-AE-i)
# ════════════════════════════════════════════════════════

def create_fl_ae(ae_name: str, origin: str) -> bool:
    """
    FL 애플리케이션 AE를 등록한다.

    rr=True는 request reachable. IN-AE가 보내는 NOTIFY를 학습 앱이 받으려면
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
        print(f"  ✓ AE created: {ae_name}  (Origin {origin})")
        return True
    if r.status_code == 409:
        print(f"  = AE already exists: {ae_name}")
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

    # 먼저 라벨로 telemetry 컨테이너를 찾는다. 이름 규칙에 기대지 않는다.
    # 'siteC:telemetry' 로 먼저 찾고, 결과마다 lbl 을 다시 확인한다.
    # 우리 컨테이너에도 site: 라벨이 붙어 있어 lbl 을 여러 개 주는 조회는
    # CSE 가 OR 로 해석하면 그것까지 로봇으로 잡는다.
    labels = [config.site_telemetry_label(node)] + config.telemetry_labels(node)
    tel_paths = om2m.discover_telemetry(
        cse, site,
        cnt_name=config.ROBOT_TELEMETRY_CNT,
        type_label=config.LABEL_TYPE_TELEMETRY,
        combined_label=config.site_telemetry_label(node),
    )

    if tel_paths:
        print(f"  Label discovery {labels}: {len(tel_paths)}/{want} robots")
        found: Dict[str, bool] = {}
        for rb in sorted(tel_paths):
            found[rb] = True
            print(f"    ✓ {tel_paths[rb]}")
        if len(found) < want:
            print(f"    ⚠ Only {len(found)} of {want} robots are labeled.")
        return found

    # 라벨이 없으면 예전처럼 이름으로 찾는다(Isaac 이 아직 라벨을 안 다는 경우).
    codes = ", ".join(str(c) for c in om2m.LAST_DISCOVERY_STATUS)
    print(f"  ⓘ Nothing found with labels {labels} (response {codes or 'none'}). "
          f"Falling back to the naming rule")
    uril = om2m.discover(cse, ty=2, lim=200)
    robots = sorted({
        p.rstrip("/").split("/")[-1]
        for p in uril
        if p.rstrip("/").split("/")[-1].startswith(f"{site}_")
    })

    print(f"  Robot AEs ({site}_*): {len(robots)}/{want} found")

    found: Dict[str, bool] = {}
    for rb in robots:
        tel = f"{cse}/{rb}/{config.ROBOT_TELEMETRY_CNT}"
        ok = om2m.resource_exists(tel)
        found[rb] = ok
        print(f"    {'✓' if ok else '✗'} {rb}/{config.ROBOT_TELEMETRY_CNT}")

    if not robots:
        print(f"    ⚠ No {site} robot AE at all. Start Isaac Sim first.")
    elif len(robots) < want:
        print(f"    ⚠ Only {len(robots)} of {want} robots are registered.")

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
    print("\n=== IN-CSE resources ===")
    cse = bind_in()

    create_fl_ae(config.IN_AE_NAME, ORIGIN_IN_AE)
    time.sleep(SETTLE)

    # 사업장 학습 앱은 IN-CSE 에도 AE 로 등록돼 있어야 한다.
    #
    # <pollingChannel> 은 AE 의 자식이고 그 AE 본인만 만들 수 있다(관리자도
    # 안 된다). 캠퍼스 방화벽이 인바운드를 버려서 IN-CSE 가 우리를 직접
    # 부르지 못하므로, 라운드 명령은 전부 그 채널로 온다. 여기에 AE 가
    # 없으면 MN 은 명령을 영영 못 받는다. 드롭박스 ACP 의 acor 도 이
    # AE-ID 를 가리킨다.
    #
    # 지금까지 이 코드가 없었다. IN-CSE 에 있던 AE 들은 예전에 손으로
    # 만든 것이고, 그 CSE 가 초기화되면 조용히 전부 멈췄을 것이다.
    for node in nodes:
        create_fl_ae(config.training_ae_name(node),
                     config.training_ae_id(node))
        time.sleep(SETTLE)

    ae = f"{cse}/{config.IN_AE_NAME}"

    # 라벨을 달아 둔다. 경로를 박는 대신 ?fu=1&lbl=... 로 찾게 하는
    # 열쇠다. type: 은 무엇인지, scope:in 은 중앙 리소스라는 뜻이다.
    def cnt(parent: str, name: str, mni: int, acp_key: Optional[str],
            kind: Optional[str] = None) -> None:
        om2m.create_container(
            parent, name, mni=mni,
            acpi=[acps[acp_key]] if acp_key and acps.get(acp_key) else None,
            labels=["scope:in", f"type:{kind or name.removeprefix('cnt-')}"],
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
            labels=["scope:in", "type:local-updates",
                    f"node:{node}", config.site_label(node)],
        )
        print(f"    ✓ cnt-local-updates/cnt-{node}")
        time.sleep(SETTLE)

    # 판정 결과. FL은 명령을 내리지 않고 여기에 써 두기만 한다.
    # 샌드박스/Isaac이 읽어서 2차 로봇 투입·대기장소 이동을 수행한다.
    cnt(ae, "cnt-fl-inference", 200, None)


# ════════════════════════════════════════════════════════
# MN-CSE 리소스
# ════════════════════════════════════════════════════════

def robot_ae_ids(cse: str, node: str) -> List[str]:
    """
    이 사업장 로봇 AE 들의 AE-ID 를 읽는다.

    추측하지 않고 CSE 에 물어본다. 로봇 AE 는 Isaac Sim 이 만들고 AE-ID 는
    CSE 가 배정하므로, 이름으로 유추하면 틀린다. ACP 의 acor 에 들어갈 값이라
    한 글자만 달라도 아이작심이 구독을 못 건다.
    """
    ids: List[str] = []
    for robot in config.robot_names(node):
        res = om2m.get_resource(f"{cse}/{robot}")
        aei = ((res or {}).get("m2m:ae") or {}).get("aei")
        if aei:
            ids.append(aei)
    return ids


def setup_analysis_ae(node: str, cse: str) -> None:
    """
    분석 앱 AE 와 판정 결과 컨테이너를 만든다.

    구조
      <MN-CSE>/ANALYSIS-AE-i/cnt-verdict/<cin>

    권한
      분석 앱   전권 (자기 리소스)
      판정 앱   RETRIEVE + NOTIFY + DISCOVERY (+ CREATE)  -> 구독해서 받는다
      로봇 AE   RETRIEVE + NOTIFY + DISCOVERY             -> 구독해서 받는다
      학습 앱   RETRIEVE + DISCOVERY                      -> 대시보드가 읽는다

    로봇에 CREATE 를 주지 않는 것이 요점이다. 판정은 분석 앱만 쓴다.
    판정 앱(CJUDGE-AE-i)에만 CREATE 를 같이 주는데, <subscription> 생성에
    CREATE 가 필요한 CSE 가 있어서다. config.VERDICT_SUB_CREATE 참고.
    """
    ae_name = config.analysis_ae_name(node)
    ae_id = config.analysis_ae_id(node)

    if not create_fl_ae(ae_name, ae_id):
        print(f"  ✗ Failed to create {ae_name} — skipping verdict containers")
        return
    time.sleep(SETTLE)

    robots = robot_ae_ids(cse, node)
    if robots:
        print(f"  Robot AE-IDs: {', '.join(robots)}")
    else:
        print(f"  ⚠ Could not read robot AE-IDs. If Isaac Sim is not up yet, "
              f"robots will not get read permission.")

    pv = [
        {"acor": [config.ORIGINATOR, ae_id], "acop": ACOP_ALL},
        {"acor": [origin_mn(node)],
         "acop": ACOP_RETRIEVE | ACOP_DISCOVERY},
    ]

    # 판정 앱. 구독을 걸어 알림으로 받아 간다.
    subs = config.verdict_subscribers(node)
    if subs:
        acop = ACOP_RETRIEVE | ACOP_NOTIFY | ACOP_DISCOVERY
        if getattr(config, "VERDICT_SUB_CREATE", True):
            acop |= ACOP_CREATE
        pv.append({"acor": subs, "acop": acop})
        print(f"  Verdict subscribers: {', '.join(subs)}  (acop={acop})")

    if robots:
        pv.append({"acor": robots,
                   "acop": ACOP_RETRIEVE | ACOP_NOTIFY | ACOP_DISCOVERY})

    acp_ri = om2m.create_acp(
        cse, f"acp-verdict-{node}", pv,
        [{"acor": [config.ORIGINATOR, ae_id], "acop": ACOP_ALL}],
    )
    print(f"  {'✓' if acp_ri else '✗'} acp-verdict-{node}")
    time.sleep(SETTLE)

    om2m.create_container(
        f"{cse}/{ae_name}", config.VERDICT_CNT,
        mni=config.VERDICT_MNI,
        acpi=[acp_ri] if acp_ri else None,
        labels=config.resource_labels(node, "verdict"),
    )
    print(f"  ✓ {ae_name}/{config.VERDICT_CNT}  (mni={config.VERDICT_MNI})")
    time.sleep(SETTLE)


def setup_mn_resources(node: str) -> Dict[str, bool]:
    """
    한 사업장 분. MEP의 MN-CSE에 바인딩하고
      1) 로봇 AE / telemetry 확인 (Isaac 소유)
      2) TRAINING-AE-i 생성 (우리 소유)
      3) cnt-local-model, cnt-net-status 생성
    """
    site = config.SITE_OF_NODE.get(node, node)
    mep = config.MEP_OF_NODE.get(node, "?")
    print(f"\n=== {node} / {site} / {mep} ===")

    cse = bind_mn(node)
    if not reachable(f"{mep} MN-CSE"):
        return {}

    robots = verify_robot_aes(node)

    ae_name = config.training_ae_name(node)
    create_fl_ae(ae_name, origin_mn(node))
    time.sleep(SETTLE)

    acp_ri = setup_mn_acp(node)
    ae = f"{cse}/{ae_name}"

    # 라운드별 로컬 모델 캐시. mni=5면 최근 5라운드만 남는다.
    om2m.create_container(
        ae, "cnt-local-model", mni=5,
        acpi=[acp_ri] if acp_ri else None,
        labels=config.resource_labels(node, "local-model"),
    )
    print("  ✓ cnt-local-model  (mni=5)")
    time.sleep(SETTLE)

    # MEC028 스냅샷과 버퍼 상태. 시나리오 2·3이 여기에 기록한다.
    om2m.create_container(
        ae, "cnt-net-status", mni=100,
        acpi=[acp_ri] if acp_ri else None,
        labels=config.resource_labels(node, "net-status"),
    )
    print("  ✓ cnt-net-status  (mni=100)")
    time.sleep(SETTLE)

    setup_analysis_ae(node, cse)

    return robots


# ════════════════════════════════════════════════════════
# Isaac Sim 대기
# ════════════════════════════════════════════════════════

def count_ready_robots(node: str) -> tuple:
    """
    이 사업장에서 telemetry 컨테이너까지 갖춘 로봇 수를 센다.
    반환: (준비된 로봇 이름들, 기대 대수)

    AE만 있고 telemetry가 없으면 아직 발행 준비가 안 된 것이므로 세지 않는다.
    """
    cse = om2m.cse_name()
    site = config.SITE_OF_NODE.get(node, node)
    want = config.ROBOTS_PER_NODE.get(node, 0)

    uril = om2m.discover(cse, ty=2, lim=200)
    names = sorted({
        p.rstrip("/").split("/")[-1]
        for p in uril
        if p.rstrip("/").split("/")[-1].startswith(f"{site}_")
    })
    ready = [n for n in names
             if om2m.resource_exists(f"{cse}/{n}/{config.ROBOT_TELEMETRY_CNT}")]
    return ready, want


def wait_for_robots(nodes: List[str],
                    timeout: float,
                    min_per_site: int) -> bool:
    """
    Isaac Sim이 로봇 AE와 telemetry를 올릴 때까지 기다린다.

    로봇 AE는 우리가 만들지 않는다. 샌드박스가 뜬 직후에는 MN-CSE만 있고
    AE는 Isaac이 등록한다. 그래서 FL을 바로 시작하면 학습할 데이터가 없다.

    기대 대수를 다 못 채워도 사업장마다 min_per_site 대만 준비되면 진행한다.
    로봇 한 대가 고장나 있다고 데모 전체를 막을 이유는 없다.
    시간이 다 되면 현재 상태를 알리고 그대로 진행한다 — 멈추지 않는다.
    """
    print(f"\n=== Waiting for Isaac Sim (up to {timeout:.0f}s, "
          f"at least {min_per_site} robots per site) ===")

    deadline = time.time() + timeout
    next_log = 0.0

    while True:
        status = {}
        for node in nodes:
            bind_mn(node)
            status[node] = count_ready_robots(node)

        done = all(len(r) >= min_per_site for r, _ in status.values())
        now = time.time()

        if done or now >= deadline:
            for node, (ready, want) in status.items():
                site = config.SITE_OF_NODE.get(node, node)
                mark = "✓" if len(ready) >= min_per_site else "✗"
                print(f"  {mark} {site}: {len(ready)}/{want} robots "
                      f"({', '.join(ready) if ready else 'none'})")
            if done:
                print("  ✓ Ready to start training")
                return True
            print(f"\n  ⚠ Not everything was ready within {timeout:.0f}s.")
            print("    Check that Isaac Sim is registering with this sandbox.")
            return False

        if now >= next_log:
            parts = [f"{config.SITE_OF_NODE.get(n, n)} {len(r)}/{w}"
                     for n, (r, w) in status.items()]
            print(f"  ⏳ {' / '.join(parts)}  "
                  f"({deadline - now:.0f}s left)")
            next_log = now + 10.0

        time.sleep(5.0)


# ════════════════════════════════════════════════════════
# 확인
# ════════════════════════════════════════════════════════

def verify(nodes: List[str]) -> bool:
    print("\n" + "=" * 60)
    print("  Resource check")
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

        ae_name = config.training_ae_name(node)
        for t in (f"{cse}/{ae_name}",
                  f"{cse}/{ae_name}/cnt-local-model",
                  f"{cse}/{ae_name}/cnt-net-status"):
            hit = om2m.resource_exists(t)
            ok &= hit
            print(f"  {'✓' if hit else '✗'} {t.split('/', 1)[-1]}")

        # 로봇은 Isaac 소유라 없어도 실패로 치지 않는다. 경고만.
        robots = verify_robot_aes(node)
        if not robots:
            print("    (training cannot run without robot data)")

    print()
    print("✓ All our resources are present" if ok else "✗ Some resources are missing")
    return ok


# ════════════════════════════════════════════════════════
# 초기화 (우리 리소스만)
# ════════════════════════════════════════════════════════

def clean(nodes: List[str]) -> None:
    """
    oneM2M DELETE로 우리가 만든 것만 지운다.
    로봇 AE는 Isaac 소유이므로 건드리지 않는다.
    """
    print("\n=== Deleting our resources ===")

    cse = bind_in()
    for name in (f"{cse}/{config.IN_AE_NAME}",):
        print(f"  {'✓' if om2m.delete_resource(name) else '✗'} {name}")
    for node in nodes:
        for n in (f"acp-dropbox-{node}",):
            om2m.delete_resource(f"{cse}/{n}")
    for n in ("acp-fl-control", "acp-global-model", "acp-local-updates"):
        om2m.delete_resource(f"{cse}/{n}")
    print("  ✓ IN-CSE ACP deleted")

    for node in nodes:
        cse = bind_mn(node)
        ae_name = config.training_ae_name(node)
        print(f"  {'✓' if om2m.delete_resource(f'{cse}/{ae_name}') else '✗'} "
              f"{config.MEP_OF_NODE.get(node)}/{ae_name}")
        om2m.delete_resource(f"{cse}/acp-cache-{node}")


# ════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true",
                    help="delete our resources and recreate them")
    ap.add_argument("--verify", action="store_true",
                    help="check only, do not create")
    ap.add_argument("--nodes", default=",".join(config.ALL_NODES),
                    help="target nodes (default: mn1,mn2,mn3)")
    ap.add_argument("--wait-robots", nargs="?", type=float,
                    const=float(os.getenv("FL_ROBOT_WAIT_SEC", "300")),
                    default=None,
                    help="wait until Isaac Sim registers the robot AEs (seconds)")
    ap.add_argument("--min-robots", type=int,
                    default=int(os.getenv("FL_MIN_ROBOTS_PER_SITE", "1")),
                    help="minimum robots per site (default 1)")
    args = ap.parse_args()

    config.require_sandbox()

    nodes = [n.strip() for n in args.nodes.split(",") if n.strip()]

    print("=" * 60)
    print("  oneM2M FL resource setup (ACME / one CSE per MEP)")
    print("=" * 60)
    print(f"  Sandbox      : {config.SANDBOX_HOST} / {config.SANDBOX_NAME}")
    print(f"  Nodes        : {', '.join(nodes)}")
    print(f"  Training     : {', '.join(config.TRAINING_NODES)}")
    print(f"  Not training : {', '.join(config.NON_CONTRIBUTING_NODES)}")

    # 먼저 4개 CSE가 다 뜨는지 본다. 하나라도 못 뜨면 생성이 반쯤
    # 끝난 상태로 남아 더 헷갈린다.
    print("\n=== CSE connectivity check ===")
    bind_in()
    alive = reachable("IN-CSE")
    for node in nodes:
        bind_mn(node)
        alive &= reachable(f"{config.MEP_OF_NODE.get(node)} MN-CSE")

    if not alive:
        print("\n✗ Some CSEs are unreachable. Check that the sandbox scenario is up")
        print("  and that FL_SANDBOX_HOST / FL_SANDBOX_NAME are correct.")
        sys.exit(1)

    if args.wait_robots is not None:
        ok = wait_for_robots(nodes, args.wait_robots, args.min_robots)
        if args.verify:
            sys.exit(0 if (verify(nodes) and ok) else 1)
        sys.exit(0 if ok else 1)

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
    print("\nNext: python3 fl/in_ae_standard.py  /  python3 fl/mn_ae_standard.py <node>")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
