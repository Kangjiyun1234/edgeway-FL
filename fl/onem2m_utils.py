"""
oneM2M Utility Functions (ACME CSE, Mca/Mcc)
- AE / CNT / CIN CRUD
- ACP (Access Control Policy) 생성
- Subscription
- Discovery with filterCriteria-like query (label-based)
- Retry: timeout/connection 에러 시 지수 백오프 재시도
"""

from __future__ import annotations

import itertools
import json
import os
import time
import requests
from typing import Dict, List, Optional, Any
import config


# -------------------------
# Retry 설정
# -------------------------
MAX_RETRY     = 4
RETRY_DELAY   = 3.0
RETRY_BACKOFF = 1.5

# ─────────────────────────────────────────────────────────
# 엔드포인트
#
# 사업장마다 MN-CSE가 따로다(mep-a/b/c). 프로세스 하나는 사업장 하나를
# 담당하므로, 기동할 때 set_endpoint()로 자기 CSE에 바인딩한다.
# IN-AE는 중앙 CSE로 바인딩한다.
# ─────────────────────────────────────────────────────────

# 이 프로세스의 '기본' CSE. MN-AE는 자기 사업장 CSE, IN-AE는 중앙 CSE다.
_BASE_URL = config.BASE_URL
_CSE_NAME = config.CSE_NAME

# 중앙 CSE. MN-AE가 FL 제어·드롭박스에 접근할 때 쓴다.
_IN_BASE_URL = config.IN_BASE_URL
_IN_CSE_NAME = config.IN_CSE_NAME

# 경로가 이 접두사로 시작하면 중앙 CSE로 보낸다.
# 사업장 CSE와 중앙 CSE 주소가 다르므로 어느 쪽인지 표시가 필요하다.
IN_PREFIX = "@in/"


def set_endpoint(base_url: str, cse_name: str) -> None:
    """이 프로세스의 기본 CSE를 지정한다."""
    global _BASE_URL, _CSE_NAME
    _BASE_URL = base_url.rstrip("/")
    _CSE_NAME = cse_name
    print(f"  [oneM2M] 기본 CSE -> {_BASE_URL}/{_CSE_NAME}")


def set_in_endpoint(base_url: str, cse_name: str) -> None:
    """중앙 CSE를 지정한다. @in/ 으로 시작하는 경로가 이쪽으로 간다."""
    global _IN_BASE_URL, _IN_CSE_NAME
    _IN_BASE_URL = base_url.rstrip("/")
    _IN_CSE_NAME = cse_name
    print(f"  [oneM2M] 중앙 CSE -> {_IN_BASE_URL}/{_IN_CSE_NAME}")


def base_url() -> str:
    return _BASE_URL


def cse_name() -> str:
    return _CSE_NAME


def in_cse_name() -> str:
    return _IN_CSE_NAME


def in_path(relative: str) -> str:
    """중앙 CSE 안의 경로를 만든다. 예: in_path("IN-AE/cnt-fl-control")"""
    return f"{IN_PREFIX}{_IN_CSE_NAME}/{relative.lstrip('/')}"


def _url(path: str) -> str:
    """리소스 경로 -> 전체 URL. @in/ 접두사면 중앙 CSE로 보낸다."""
    p = str(path).lstrip("/")
    if p.startswith(IN_PREFIX):
        return f"{_IN_BASE_URL}/{p[len(IN_PREFIX):]}"
    return f"{_BASE_URL}/{p}"


def strip_prefix(path: str) -> str:
    """로그 출력용. @in/ 표시를 떼어낸다."""
    p = str(path).lstrip("/")
    return p[len(IN_PREFIX):] if p.startswith(IN_PREFIX) else p


def _verify():
    return getattr(config, "TLS_VERIFY", True)


def _auth_for(url: str):
    """
    이 URL이 어느 CSE로 가는지 보고 HTTP 기본 인증 정보를 고른다.

    샌드박스의 MN-CSE는 열려 있지만 클라우드 IN-CSE는 401을 돌려준다.
    두 CSE의 자격증명이 다를 수 있으므로 URL로 구분한다.
    설정이 없으면 None이고, requests는 auth=None을 인증 없음으로 처리한다.
    """
    try:
        if _IN_BASE_URL and url.startswith(_IN_BASE_URL):
            return config.in_auth()
        return config.mn_auth()
    except Exception:
        return None


def _request_with_retry(method: str, url: str, **kwargs) -> requests.Response:
    delay = RETRY_DELAY
    last_exc = None
    kwargs.setdefault("verify", _verify())
    kwargs.setdefault("auth", _auth_for(url))
    for attempt in range(MAX_RETRY):
        try:
            return requests.request(method, url, **kwargs)
        except requests.exceptions.SSLError as e:
            # 샌드박스가 IP 기반 자체 서명 인증서를 쓰면 여기서 막힌다.
            print(f"  ✗ TLS 검증 실패: {url}")
            print("    자체 서명 인증서라면 FL_TLS_VERIFY=0 으로 실행하라.")
            raise e
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_exc = e
            if attempt < MAX_RETRY - 1:
                print(f"  ⏳ retry {attempt+1}/{MAX_RETRY-1} ({delay:.1f}s): {type(e).__name__}")
                time.sleep(delay)
                delay *= RETRY_BACKOFF
    raise last_exc


# -------------------------
# Headers
# -------------------------

_RI_COUNTER = itertools.count(1)


def _next_request_id() -> str:
    """
    X-M2M-RI (Request Identifier).

    TS-0009 HTTP 바인딩에서 모든 요청에 필수이며 요청마다 고유해야 한다.
    기존 구현은 공통 헤더에 아예 넣지 않았다. 관대한 CSE에서는 통과하지만
    표준 적합성을 보는 쪽에서는 가장 먼저 지적되는 항목이다.
    """
    return f"{os.getpid()}-{time.time_ns()}-{next(_RI_COUNTER)}"


def _headers(ty: Optional[int] = None) -> Dict[str, str]:
    h = config.HEADERS.copy()
    h["X-M2M-RI"] = _next_request_id()
    if ty is not None:
        h["Content-Type"] = f"application/json;ty={ty}"
    return h


# -------------------------
# ACP
# -------------------------

def create_acp(
    parent_path: str,
    acp_name: str,
    pv_rules: List[Dict],
    pvs_rules: List[Dict],
) -> Optional[str]:
    """
    ACP 생성 후 ri 반환.
    pv_rules/pvs_rules 예시:
      [{"acor": ["CAdmin", "CIN-AE"], "acop": 63}]
    """
    url = _url(parent_path)
    payload = {
        "m2m:acp": {
            "rn": acp_name,
            "pv":  {"acr": pv_rules},
            "pvs": {"acr": pvs_rules},
        }
    }
    try:
        r = _request_with_retry("post", url, json=payload, headers=_headers(1), timeout=10)
        if r.status_code == 201:
            ri = r.json().get("m2m:acp", {}).get("ri")
            return ri
        elif r.status_code == 409:
            # 이미 존재 → ri 조회
            get_url = _url(f"{parent_path}/{acp_name}")
            rg = _request_with_retry("get", get_url, headers=_headers(), timeout=10)
            if rg.status_code == 200:
                return rg.json().get("m2m:acp", {}).get("ri")
        print(f"✗ ACP create failed: {r.status_code} - {r.text}")
        return None
    except Exception as e:
        print(f"✗ ACP create error: {e}")
        return None


def update_acpi(resource_path: str, acpi: List[str]) -> bool:
    """
    기존 컨테이너에 acpi 연결 (PUT/UPDATE).
    acpi: ACP ri 목록 e.g. ["1-20260309T221953xxxx"]
    """
    url = _url(resource_path)
    payload = {"m2m:cnt": {"acpi": acpi}}
    try:
        r = _request_with_retry("put", url, json=payload, headers=_headers(), timeout=10)
        return r.status_code in (200, 204)
    except Exception as e:
        print(f"✗ update_acpi error: {e}")
        return False


# -------------------------
# AE / CNT / CIN
# -------------------------

def create_application(ae_name: str) -> Optional[Dict]:
    url = _url(_CSE_NAME)
    payload = {
        "m2m:ae": {
            "rn": ae_name,
            "api": f"N.{ae_name}.app",
            "rr": True,
            "srv": ["2a"],
        }
    }
    try:
        r = _request_with_retry("post", url, json=payload, headers=_headers(2), timeout=10)
        if r.status_code in (201, 409):
            return r.json() if r.status_code == 201 else {"exists": True}
        print(f"✗ AE create failed: {r.status_code} - {r.text}")
        return None
    except Exception as e:
        print(f"✗ AE create error: {e}")
        return None


def delete_resource(resource_path: str) -> bool:
    """리소스 삭제 (존재하지 않으면 True 반환)."""
    url = _url(resource_path)
    try:
        r = _request_with_retry("delete", url, headers=_headers(), timeout=10)
        return r.status_code in (200, 202, 204, 404)
    except Exception as e:
        print(f"✗ DELETE error: {e}")
        return False


def create_container(
    parent_path: str,
    container_name: str,
    mni: int = 1000,
    mbs: int = 10_000_000,
    acpi: Optional[List[str]] = None,
) -> Optional[Dict]:
    """
    Container 생성.
    mbs: max byte size (기본 10MB — 가중치 CIN이 들어갈 여유)
    acpi: ACP ri 목록. 지정 시 생성과 동시에 acpi 연결 시도,
          실패하면 생성 후 UPDATE로 재시도.
    """
    url = _url(parent_path)
    cnt_obj: Dict[str, Any] = {"rn": container_name, "mni": mni, "mbs": mbs}
    if acpi:
        cnt_obj["acpi"] = acpi

    payload = {"m2m:cnt": cnt_obj}
    try:
        r = _request_with_retry("post", url, json=payload, headers=_headers(3), timeout=10)
        if r.status_code in (201, 409):
            result = r.json() if r.status_code == 201 else {"exists": True}

            # acpi가 있는데 생성 응답에 acpi가 없으면 UPDATE로 재시도
            if acpi:
                created_acpi = (r.json().get("m2m:cnt", {}).get("acpi", [])
                                if r.status_code == 201 else [])
                if not created_acpi:
                    resource_path = f"{parent_path}/{container_name}"
                    ok = update_acpi(resource_path, acpi)
                    if not ok:
                        print(f"  ⚠ acpi UPDATE 실패: {resource_path}")

            return result
        print(f"✗ CNT create failed: {r.status_code} - {r.text}")
        return None
    except Exception as e:
        print(f"✗ CNT create error: {e}")
        return None


def create_content_instance(
    container_path: str,
    content: Any,
    labels: Optional[List[str]] = None,
    resource_name: Optional[str] = None,
) -> Optional[Dict]:
    """
    resource_name을 주면 CIN의 rn을 직접 지정한다.
    가중치 조각처럼 수신측이 discovery 없이 이름으로 바로 RETRIEVE 해야 하는
    경우에 쓴다.
    """
    url = _url(container_path)
    con_str = (json.dumps(content, separators=(",", ":"), ensure_ascii=False)
               if isinstance(content, (dict, list)) else str(content))

    cin_obj: Dict[str, Any] = {"con": con_str}

    # contentInfo. con에 무엇이 들어 있는지 리소스만 보고 알 수 있게 한다.
    # dict/list를 JSON 문자열로 직렬화해 넣으므로 application/json:0 이다.
    cnf = getattr(config, "CIN_CONTENT_INFO", None)
    if cnf and isinstance(content, (dict, list)):
        cin_obj["cnf"] = cnf

    if resource_name:
        cin_obj["rn"] = resource_name
    if labels:
        cin_obj["lbl"] = labels

    payload: Dict[str, Any] = {"m2m:cin": cin_obj}

    try:
        r = _request_with_retry("post", url, json=payload, headers=_headers(4), timeout=45)
        if r.status_code == 201:
            return r.json()
        print(f"✗ CIN create failed: {r.status_code} - {r.text}")
        return None
    except Exception as e:
        print(f"✗ CIN create error: {e}")
        return None


def get_latest_content_instance(container_path: str) -> Optional[Dict]:
    url = _url(f"{container_path}/la")
    try:
        r = _request_with_retry("get", url, headers=_headers(), timeout=10)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception as e:
        print(f"✗ get_latest CIN error: {e}")
        return None


def get_resource(resource_path: str) -> Optional[Dict]:
    url = _url(resource_path)
    try:
        r = _request_with_retry("get", url, headers=_headers(), timeout=10)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception as e:
        print(f"✗ get_resource error: {e}")
        return None


# -------------------------
# Subscription
# -------------------------

def create_subscription(
    parent_path: str,
    subscription_name: str,
    notification_uri: str,
    event_types: List[int],
    use_nct: Optional[int] = None,
) -> Optional[Dict]:
    url = _url(parent_path)
    sub_obj: Dict[str, Any] = {
        "rn": subscription_name,
        "nu": [notification_uri],
        "enc": {"net": event_types},
    }
    if use_nct is not None:
        sub_obj["nct"] = use_nct

    payload = {"m2m:sub": sub_obj}
    try:
        r = _request_with_retry("post", url, json=payload, headers=_headers(23), timeout=10)
        if r.status_code in (201, 409):
            return r.json() if r.status_code == 201 else {"exists": True}
        print(f"✗ SUB create failed: {r.status_code} - {r.text}")
        return None
    except Exception as e:
        print(f"✗ SUB create error: {e}")
        return None


def delete_subscription(parent_path: str, subscription_name: str) -> bool:
    url = _url(f"{parent_path}/{subscription_name}")
    try:
        r = _request_with_retry("delete", url, headers=_headers(), timeout=10)
        return r.status_code in (200, 404)
    except Exception:
        return False


# -------------------------
# Discovery (label-based)
# -------------------------

def discover_uril_by_label(container_path: str, label: str, lim: int = 20) -> List[str]:
    base = _url(container_path)
    candidates = [
        f"{base}?fu=1&ty=4&lim={lim}&lbl={label}",
        f"{base}?fu=1&ty=4&lim={lim}&lbl={requests.utils.quote(label)}",
        f"{base}?fu=1&lbl={label}",
    ]
    for url in candidates:
        try:
            r = _request_with_retry("get", url, headers=_headers(), timeout=10)
            if r.status_code == 200:
                data = r.json()
                uril = data.get("m2m:uril", [])
                if isinstance(uril, list) and uril:
                    return uril
        except Exception:
            continue
    return []


def list_uril(container_path: str, lim: int = 50) -> List[str]:
    url = _url(f"{container_path}?fu=1&ty=4&lim={lim}")
    try:
        r = _request_with_retry("get", url, headers=_headers(), timeout=10)
        if r.status_code != 200:
            return []
        data = r.json()
        uril = data.get("m2m:uril", [])
        return uril if isinstance(uril, list) else []
    except Exception:
        return []


def discover(parent_path: str, ty: int, lim: int = 100) -> List[str]:
    """
    자손 리소스 디스커버리 (fu=1). ty로 종류를 고른다.
      ty=2 AE, ty=3 container, ty=4 contentInstance

    반환은 구조화된 리소스가 아니라 경로 문자열 목록(m2m:uril)이다.
    Isaac Sim이 만들어 둔 로봇 AE와 그 telemetry 컨테이너가
    실제로 있는지 확인하는 데 쓴다.
    """
    url = _url(f"{parent_path}?fu=1&ty={ty}&lim={lim}")
    try:
        r = _request_with_retry("get", url, headers=_headers(), timeout=15)
        if r.status_code != 200:
            return []
        uril = r.json().get("m2m:uril", [])
        return uril if isinstance(uril, list) else []
    except Exception as e:
        print(f"✗ discover error: {e}")
        return []


def resource_exists(resource_path: str) -> bool:
    """단건 RETRIEVE로 존재 여부만 확인한다."""
    return get_resource(resource_path) is not None


def retrieve_cins_by_label(
    container_path: str,
    label: str,
    lim: int = 30,
    fallback_scan: int = 120,
    newest_first: bool = True,
) -> List[Dict]:
    def _ct(res: Dict) -> str:
        try:
            return (res.get("m2m:cin") or {}).get("ct", "") or ""
        except Exception:
            return ""

    matched: List[Dict] = []

    uril = discover_uril_by_label(container_path, label, lim=lim)
    if uril:
        for u in uril:
            cin = get_resource(u.lstrip("/"))
            if not cin or "m2m:cin" not in cin:
                continue
            if label in (cin["m2m:cin"].get("lbl") or []):
                matched.append(cin)
        if matched:
            matched.sort(key=_ct, reverse=newest_first)
            return matched

    uril2 = list_uril(container_path, lim=fallback_scan)
    if not uril2:
        return []

    for u in reversed(uril2):
        cin = get_resource(u.lstrip("/"))
        if not cin or "m2m:cin" not in cin:
            continue
        if label in (cin["m2m:cin"].get("lbl") or []):
            matched.append(cin)

    matched.sort(key=_ct, reverse=newest_first)
    return matched


def retrieve_latest_cin_by_label(
    container_path: str,
    label: str,
    lim: int = 30,
    fallback_scan: int = 80,
) -> Optional[Dict]:
    cins = retrieve_cins_by_label(
        container_path, label, lim=lim, fallback_scan=fallback_scan
    )
    return cins[0] if cins else None


def retrieve_first_cin_by_label(container_path: str, label: str) -> Optional[Dict]:
    return retrieve_latest_cin_by_label(container_path, label)