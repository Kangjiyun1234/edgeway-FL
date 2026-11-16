"""
fl/onem2m_pch.py — oneM2M <pollingChannel> 알림 수신

[이름에 대하여]
pch 는 oneM2M 이 <pollingChannel> 에 붙인 공식 약어다(m2m:pch, ty=15).
"폴링"이라는 일반 명사로 부르면 "주기적으로 긁는다"로 읽히는데,
이건 그것과 다르다. 구독도 알림도 그대로 두고 전달 방향만 뒤집는다.
규격 용어를 그대로 쓰는 편이 오해가 없다.

[왜 필요한가]
클라우드 IN-CSE(45.79.142.139)가 방화벽 뒤의 우리 AE를 호출하지 못한다.
구독을 만들 때 CSE가 알림 주소로 검증 요청을 보내는데, 그게 막혀서
"Verification request failed" 또는 Read timeout 으로 끝난다.

TS-0001은 이런 AE를 "요청 도달 불가(request-unreachable)"로 보고
<pollingChannel>을 쓰도록 정의한다. 우회가 아니라 규격에 있는 길이다.

  전:  CSE ──POST /notify──▶ AE          방화벽이 버림
  후:  AE ──GET pcu(롱폴링)──▶ CSE
       CSE ──알림을 그 응답으로──▶ AE     AE가 건 연결이라 막히지 않음

구독도 알림 의미도 그대로다. 전달 경로만 바뀐다.

[동작]
  1) AE 아래에 <pollingChannel>(ty=15)을 만든다
  2) AE를 rr=false 로 바꿔 "직접 호출하지 말라"고 CSE에 알린다
  3) 구독의 nu 를 URL 대신 AE-ID 로 준다 (예: ["CIN-AE"])
  4) 이 모듈이 {ae}/{pch}/pcu 를 계속 롱폴링한다

[실측한 것 — ACME 기준]
  큐가 비면            504 + {"m2m:dbg": "Request Expiration Timestamp reached.
                              No request queued for originator: CIN-AE"}
                       에러가 아니다. 다시 걸면 된다.
  알림이 있으면        200 + {"m2m:rqp": {"fr","to","op":5,"rqi","pc":{"m2m:sgn":{...}}}}
  pc 안의 모양         기존 HTTP /notify 가 받던 본문과 동일하다.
                       그래서 알림 처리 로직을 그대로 재사용한다.
  큐 단위              originator(AE-ID) 별이다.
"""

import json
import os
import threading
import time
from typing import Callable, Optional

import config
import onem2m_utils as om2m


# 롱폴링 한 번이 CSE에서 열려 있는 시간. ACME 기본이 10초쯤이다.
# 너무 짧으면 재요청이 잦고, 너무 길면 종료가 굼뜨다.
PCH_WAIT_SEC = float(os.getenv("FL_PCH_WAIT_SEC", "20"))

# 연속 실패 시 쉬는 시간. CSE가 잠깐 흔들릴 때 폭주하지 않게 한다.
PCH_BACKOFF_SEC = float(os.getenv("FL_PCH_PCH_BACKOFF_SEC", "3"))

DEFAULT_NAME = os.getenv("FL_PCH_NAME", "pch")


def ensure_channel(ae_path: str, name: str = DEFAULT_NAME) -> bool:
    """
    AE 아래에 <pollingChannel>을 만든다. 이미 있으면 그대로 둔다.

    ae_path 는 CSE 이름까지 포함한 경로다. 예: "estimed-in/IN-AE"
    """
    full = f"{ae_path}/{name}"
    if om2m.resource_exists(full):
        print(f"  = pollingChannel 있음: {om2m.strip_prefix(full)}")
        return True

    url = om2m._url(ae_path)
    try:
        r = om2m._request_with_retry(
            "post", url,
            json={"m2m:pch": {"rn": name}},
            headers=om2m._headers(15),
            timeout=10,
        )
    except Exception as e:
        print(f"  ✗ pollingChannel 생성 오류: {e}")
        return False

    if r.status_code in (201, 409):
        print(f"  ✓ pollingChannel: {om2m.strip_prefix(full)}")
        return True
    print(f"  ✗ pollingChannel 생성 실패: {r.status_code} {r.text[:160]}")
    return False


def set_request_unreachable(ae_path: str, unreachable: bool = True) -> bool:
    """
    AE의 rr(requestReachability)을 바꾼다.

    rr=true 면 CSE가 AE를 직접 호출하려 한다. 방화벽 뒤에서는 그게 실패하므로
    false 로 두어야 CSE가 폴링 채널에 알림을 쌓는다.
    """
    try:
        r = om2m._request_with_retry(
            "put", om2m._url(ae_path),
            json={"m2m:ae": {"rr": not unreachable}},
            headers=om2m._headers(body=True),
            timeout=10,
        )
    except Exception as e:
        print(f"  ✗ rr 변경 오류: {e}")
        return False

    if r.status_code in (200, 204):
        print(f"  ✓ {om2m.strip_prefix(ae_path)} rr={not unreachable}")
        return True
    print(f"  ✗ rr 변경 실패: {r.status_code} {r.text[:160]}")
    return False


def relay_to_local(notify_url: str) -> Callable[[dict], None]:
    """
    받은 알림을 로컬 HTTP 핸들러로 그대로 넘기는 handler 를 만든다.

    알림 처리 로직(구독 검증, net 판정, 결과 수집)은 이미 Flask 라우트 안에
    있다. 그걸 뜯어 고치는 대신, pollingChannel 로 받은 본문을 같은 라우트에
    127.0.0.1 로 POST 한다. 전달 경로만 바뀌고 처리 코드는 그대로다.

    로컬 왕복이 한 번 늘지만 루프백이라 무시할 수준이고,
    검증된 코드를 건드리지 않는 쪽이 낫다.
    """
    import requests

    def _handler(pc: dict) -> None:
        try:
            requests.post(notify_url, json=pc, timeout=10)
        except Exception as e:
            print(f"  ⚠ 로컬 알림 전달 실패 ({notify_url}): {e}")

    return _handler


class PchReceiver(threading.Thread):
    """
    <pollingChannel>을 롱폴링해 알림을 받아 handler 로 넘기는 스레드.

    handler 는 기존 HTTP /notify 가 받던 것과 같은 dict 를 받는다.
    즉 {"m2m:sgn": {...}} 형태다. 그래서 알림 처리 로직을 고칠 필요가 없다.
    """

    def __init__(self, ae_path: str, origin: str,
                 handler: Callable[[dict], None],
                 name: str = DEFAULT_NAME):
        super().__init__(daemon=True, name=f"pch-{origin}")
        self.ae_path = ae_path
        self.pcu_path = f"{ae_path}/{name}/pcu"
        self.origin = origin
        self.handler = handler
        self._stop = threading.Event()
        self.received = 0
        self.errors = 0

    def stop(self) -> None:
        self._stop.set()

    # ── 내부 ────────────────────────────────────────────

    def _headers(self) -> dict:
        h = om2m._headers()
        h["X-M2M-Origin"] = self.origin
        return h

    def _respond(self, rqp: dict) -> None:
        """
        응답 프리미티브를 돌려보낸다.

        CSE는 알림을 보낸 뒤 응답을 기다린다. 안 보내면 그 요청이 미결로 남아
        다음 알림이 내려오지 않을 수 있다. rqi 를 그대로 돌려줘야 CSE가
        어느 요청에 대한 응답인지 안다.
        """
        rqi = rqp.get("rqi")
        if not rqi:
            return
        rsp = {
            "m2m:rsp": {
                "fr": self.origin,
                "to": rqp.get("fr", ""),
                "rqi": rqi,
                "rsc": 2000,                      # 2000 = OK
                "rvi": config.HEADERS.get("X-M2M-RVI", "3"),
            }
        }
        try:
            om2m._request_with_retry(
                "post", om2m._url(self.pcu_path),
                json=rsp, headers=self._headers(), timeout=10,
            )
        except Exception as e:
            print(f"  ⚠ [{self.origin}] 응답 프리미티브 전송 실패: {e}")

    def _handle(self, body: dict) -> None:
        rqp = body.get("m2m:rqp") or {}
        pc = rqp.get("pc") or {}

        if not pc:
            return

        self.received += 1
        try:
            self.handler(pc)
        except Exception as e:
            print(f"  ⚠ [{self.origin}] 알림 처리 중 오류: {e}")
        finally:
            # 처리가 실패해도 응답은 보낸다. 안 보내면 큐가 막힌다.
            self._respond(rqp)

    # ── 루프 ────────────────────────────────────────────

    def run(self) -> None:
        print(f"  ✓ [{self.origin}] pollingChannel 수신 시작 "
              f"({om2m.strip_prefix(self.pcu_path)})")

        while not self._stop.is_set():
            try:
                r = om2m._request_with_retry(
                    "get", om2m._url(self.pcu_path),
                    headers=self._headers(),
                    timeout=PCH_WAIT_SEC + 10,
                )
            except Exception as e:
                self.errors += 1
                if not self._stop.is_set():
                    print(f"  ⚠ [{self.origin}] 폴링 오류: {e}")
                    time.sleep(PCH_BACKOFF_SEC)
                continue

            if r.status_code == 200:
                try:
                    self._handle(r.json())
                except json.JSONDecodeError:
                    print(f"  ⚠ [{self.origin}] 알림 파싱 실패: {r.text[:160]}")
                continue

            if r.status_code == 504:
                # 큐가 비었다. 정상이다. 바로 다시 건다.
                continue

            self.errors += 1
            print(f"  ⚠ [{self.origin}] 폴링 응답 {r.status_code}: "
                  f"{r.text[:160]}")
            time.sleep(PCH_BACKOFF_SEC)

        print(f"  [{self.origin}] pollingChannel 수신 종료 "
              f"(받은 알림 {self.received}개, 오류 {self.errors}회)")


def notification_target(ae_id: str) -> str:
    """
    구독의 nu 에 넣을 값.

    URL 대신 AE-ID 를 주면 CSE가 그 AE의 도달성을 보고 폴링 채널에 쌓는다.
    rr=false 이고 <pollingChannel>이 있어야 동작한다.
    """
    return ae_id
