"""
표준 oneM2M FL 아키텍처 - MN-AE Trainer (Raw Signal AE 버전)
AMR 주행 텔레메트리(38채널 CVAE)용

변경 사항 (이전 통계피처 버전 대비):
- AEEdgeNode(조건부 1D CNN VAE)
- 입력: raw 시계열 신호 (N, seq_len)
- 손실: MSE reconstruction loss (정상만)
- val_acc → val_auroc (재구성 오차 기반 AUROC)
"""
import sys

import os
import time
import json
import pickle
import threading
import hashlib
import shutil

import numpy as np
import torch
from flask import Flask, request, make_response

import amr_ingest
import config
import model_codec
import onem2m_utils as om2m
import onem2m_pch
import verdict
from edge_node import AEEdgeNode

# "이번 라운드는 학습할 수 없다. 다만 노드를 죽일 일은 아니다"를 나타내는 신호.
# 통신 실패(None)와 구분해야 한다. 전자는 다음 라운드에 회복될 수 있지만
# 후자는 그렇지 않다.
SKIP_ROUND = object()


class MNAETrainer:
    def __init__(self, node_id: int, notification_port: int, inject_anomaly: bool = False):
        self.node_id          = node_id
        self.node_name        = f"mn{node_id + 1}"

        # 노드 역할
        #   관측 전용 : 학습 안 하고 추론만 (COLD_START_NODES)
        #   로컬 전용 : 로컬 학습은 하되 FedAvg에 가중치를 올리지 않음
        #               (LOCAL_ONLY_NODES — 사업장 C가 여기 해당)
        #   그 외     : 학습 + FedAvg 기여
        self.is_cold_start = self.node_name in getattr(
            config, "COLD_START_NODES", []
        )
        self.is_local_only = self.node_name in getattr(
            config, "LOCAL_ONLY_NODES", []
        )
        self.notification_port = notification_port
        self.inject_anomaly   = inject_anomaly

        # 이 사업장의 정규화 통계. 첫 수집에서 잡고 이후 라운드에서 재사용한다.
        self.norm_mean = None
        self.norm_std = None

        self.ae_name = f"MN-AE-{node_id + 1}"
        self.ae_id = config.mn_ae_id(self.node_name)

        # 켜져 있으면 이 사업장의 AE-ID 로 CSE에 요청한다. 그래야 ACP가
        # 실제로 작동한다. 꺼져 있으면 기존처럼 CAdmin 으로 나간다.
        if config.USE_AE_IDENTITY:
            om2m.set_originator(self.ae_id)

        # 이 사업장의 MN-CSE에 바인딩한다. 사업장마다 MEP가 다르다.
        mn_base, mn_cse = config.mn_endpoint(self.node_name)
        om2m.set_endpoint(mn_base, mn_cse)
        om2m.set_in_endpoint(*config.in_endpoint())
        self.mn_cse_name = mn_cse

        # 사업장 CSE 안의 리소스.
        #
        # 텔레메트리는 MN-AE가 아니라 로봇 AE 아래에 쌓인다. Isaac Sim이
        # 로봇 AE를 등록하고 주행하면서 윈도우를 CREATE한다.
        #     <MN-CSE>/siteA_AMR_1/telemetry/<cin>
        # 사업장 하나가 FL 노드 하나이므로 로봇 대수만큼의 컨테이너를
        # 여기서 모아 "한 공장의 데이터"로 합친다.
        self.robot_names = config.robot_names(self.node_name)
        # 로봇별로 어디까지 읽었는지. 라운드마다 그 뒤만 읽는다.
        # 비워 두면 첫 라운드는 최근 분량을 받는다.
        self.telemetry_cursor = {}

        # 이번 라운드에 모은 윈도우. telemetry 컨테이너의 mni 때문에
        # 한 번에 다 긁을 수 없어서, 받을 때마다 여기에 쌓는다.
        self._round_buffer = None
        self._round_buffer_round = None
        self.telemetry_paths = [
            f"{mn_cse}/{rb}/{config.ROBOT_TELEMETRY_CNT}"
            for rb in self.robot_names
        ]
        self.local_model_cnt   = "cnt-local-model"
        self.local_model_path  = f"{mn_cse}/{self.ae_name}/{self.local_model_cnt}"

        # 중앙 CSE 안의 리소스. @in/ 접두사가 중앙으로 보낸다.
        self.fl_control_path   = om2m.in_path(f"{config.IN_AE_NAME}/cnt-fl-control")
        self.global_model_path = om2m.in_path(f"{config.IN_AE_NAME}/cnt-global-model")

        # Drop-box (중앙)
        self.dropbox_cnt_path = om2m.in_path(
            f"{config.IN_AE_NAME}/cnt-local-updates/cnt-{self.node_name}"
        )

        # NOTIFY 기반 round 시작
        # 텔레메트리 구독으로 받은 알림을 구분하기 위한 구독 ri 집합.
        # 라운드 명령 알림과 섞이면 안 된다.
        self.telemetry_subs = set()

        # 로봇이 새 윈도우를 올렸다는 신호. 수집 루프가 이걸 기다린다.
        # 알림이 깨우므로 2초마다 CSE를 찔러볼 필요가 없다.
        self.data_event = threading.Event()
        self.data_notifications = 0

        # 알림 경로는 둘이고 조건이 서로 다르다. 한 변수로 묶으면
        # 한쪽 설정이 다른 쪽을 망가뜨린다.
        #   local  : 사업장 MN-CSE. 로봇 텔레메트리 알림.
        #   remote : 클라우드 IN-CSE. 라운드 명령 알림.
        # 이 프로세스가 뜬 시각. 이보다 먼저 발행된 라운드 명령은
        # 지난 실행의 잔재이므로 따르지 않는다.
        self.started_at = time.time()

        self.local_notify_target = None
        self.remote_notify_target = None
        self.local_pch = None
        self.pch_receiver = None

        # 판정(분석 앱). 결과는 사업장 밖으로 안 나가므로 MN-CSE 에 남긴다.
        # 쓰는 신원은 MN-AE 가 아니라 분석 앱이다. 설계서에서 둘은 다른 앱이고
        # ACP 도 역할별로 나뉜다.
        self.analysis_ae_id = config.analysis_ae_id(self.node_name)
        self.verdict_cnt_path = (
            f"{self.mn_cse_name}/"
            f"{config.analysis_ae_name(self.node_name)}/"
            f"{config.VERDICT_CNT}"
        )
        # 이번 라운드에 읽은 윈도우. 판정은 학습이 끝난 뒤 이걸로 낸다.
        self.last_windows = None

        self.job_event = threading.Event()
        self.job_lock = threading.Lock()
        self.latest_job_command = None
        self.last_command_source = None

        # NOTIFY가 누락됐을 때만 /la를 한 번 확인하기 위한 timeout
        self.command_wait_timeout_sec = float(
            os.getenv("FL_COMMAND_NOTIFY_TIMEOUT", "30")
        )

        self.edge_node = None

        self.app = Flask(f"MN-AE-{self.node_name}")
        self._setup_flask()

        self.edge_cache_dir = f"/tmp/fl_models/cache/{self.node_name}"
        os.makedirs(self.edge_cache_dir, exist_ok=True)

        self.local_cache_meta_file = os.path.join(
            self.edge_cache_dir, "cache_meta.json"
        )

        self.round_start_stagger_sec = float(self.node_id) * 2.0
        self.upload_stagger_sec      = float(self.node_id) * 3.0

    # ---------------------------
    # Hash
    # ---------------------------
    @staticmethod
    def _sha256_file(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    # ---------------------------
    # Flask / Notify
    # ---------------------------
    def _setup_flask(self):
        @self.app.route("/notify", methods=["POST"])
        def handle_notification():
            try:
                body = request.get_json(silent=True) or {}
                sgn = body.get("m2m:sgn", {})

                # Subscription verification
                if sgn.get("vrq") is True:
                    print(
                        f"  [VERIFY] {self.node_name} "
                        f"subscription verification OK"
                    )

                    resp = make_response("", 200)
                    resp.headers["X-M2M-RSC"] = "2000"
                    return resp

                nev = sgn.get("nev", {}) or {}
                net = nev.get("net", 0)
                net_values = net if isinstance(net, list) else [net]

                try:
                    is_cin_created = 3 in {
                        int(value) for value in net_values
                    }
                except (TypeError, ValueError):
                    is_cin_created = False

                if not is_cin_created:
                    resp = make_response("", 200)
                    resp.headers["X-M2M-RSC"] = "2000"
                    return resp

                # 텔레메트리 알림이면 "새 데이터 있음"만 세우고 끝낸다.
                #
                # 알림마다 CSE를 읽으러 가면 윈도우 수만큼 왕복이 생긴다.
                # 여기서는 표시만 하고, 실제 읽기는 라운드 수집 때 한 번에
                # 한다. 알림은 "언제 읽을지"를, cra 증분 조회는 "무엇을
                # 읽을지"를 정한다.
                sur = str(sgn.get("sur", "")).strip("/").split("/")[-1]
                if sur in self.telemetry_subs:
                    self.data_notifications += 1
                    self.data_event.set()
                    resp = make_response("", 200)
                    resp.headers["X-M2M-RSC"] = "2000"
                    return resp

                # 정상 경로:
                # round command CIN의 con을 NOTIFY body에서 읽음
                command = self._parse_cin_con_from_notification(sgn)

                # CSE가 rep를 보내지 않은 경우에만
                # NOTIFY를 계기로 최신 CIN을 한 번 조회
                #
                # rep 가 없으면 그냥 무시한다. 전에는 /la 를 대신 읽었는데
                # 그게 묵은 명령으로 라운드를 시작시켰다.
                #
                # cnt-fl-control 에는 지난 실행의 CIN 이 남아 있다. 기동 직후
                # rep 없는 알림이 하나만 와도 /la 가 지난 실행의
                # FL_TRAINING round=1 을 돌려주고, MN-AE 는 그것을 이번
                # 라운드 명령으로 알고 학습을 시작한다. 실제로 그렇게 됐다.
                # IN-AE 가 아직 라운드를 열기도 전에 학습하고 업로드했고,
                # 그 업로드는 IN-AE 가 드롭박스를 비우면서 사라졌다.
                #
                # 알림을 놓친 경우는 _wait_for_round_command 의 타임아웃
                # 폴백이 이미 처리한다. 그쪽은 라운드 번호를 대조하므로 안전하다.
                if command is None:
                    resp = make_response("", 200)
                    resp.headers["X-M2M-RSC"] = "2000"
                    return resp

                if command:
                    state = str(command.get("jobState", ""))

                    try:
                        current_round = int(
                            command.get("currentRound", 0)
                        )
                    except (TypeError, ValueError):
                        current_round = 0

                    command_type = (
                        str(command.get("type", ""))
                        .lower()
                        .replace("_", "-")
                    )

                    if command_type and command_type != "fl-command":
                        print(
                            f"  ⚠ {self.node_name}: "
                            f"ignore command type={command_type!r}"
                        )

                    elif self._is_stale_command(command):
                        pass

                    elif state in ("FL_TRAINING", "FL_COMPLETED"):
                        with self.job_lock:
                            self.latest_job_command = command
                            self.last_command_source = "NOTIFY"

                        cin_ri = (
                            ((sgn.get("nev") or {}).get("rep") or {})
                            .get("m2m:cin", {})
                            .get("ri", "?")
                        )
                        print(
                            f"  [NOTIFY] {self.node_name} "
                            f"command received: "
                            f"{state}, round={current_round}, cin={cin_ri}"
                        )

                        self.job_event.set()

                resp = make_response("", 200)
                resp.headers["X-M2M-RSC"] = "2000"
                return resp

            except Exception as e:
                print(f"  ✗ notify error: {e}")
                return make_response("", 500)

    def _parse_cin_con_from_notification(self, sgn: dict):
        nev = sgn.get("nev", {}) or {}
        rep = nev.get("rep", {}) or {}

        cin = rep.get("m2m:cin")
        if not isinstance(cin, dict):
            return None

        con = cin.get("con")
        if con is None:
            return None

        try:
            data = json.loads(con) if isinstance(con, str) else con
        except (TypeError, json.JSONDecodeError):
            return None

        return data if isinstance(data, dict) else None

    def start_server(self):
        def run():
            self.app.run(
                host="0.0.0.0", port=self.notification_port,
                threaded=True, use_reloader=False
            )
        th = threading.Thread(target=run, daemon=True)
        th.start()
        time.sleep(1)
        print(f"  ✓ {self.node_name} Notification server on port {self.notification_port}")
        self._setup_notify_path()

    def _setup_notify_path(self):
        """
        알림을 어느 경로로 받을지 정한다. 경로는 둘이고 조건이 다르다.

          로컬  사업장 MN-CSE -> 우리.  로봇 텔레메트리 알림이 이 길로 온다.
                같은 사업장 안이라 CSE 가 우리를 직접 부를 수 있다.
                AE 에 poa 를 박고 구독의 nu 에 URL 을 준다.

          원격  클라우드 IN-CSE -> 우리.  라운드 명령(cnt-fl-control)이 이 길.
                캠퍼스 방화벽이 인바운드를 버리므로 CSE 가 우리를 못 부른다.
                <pollingChannel> 로 받고 구독의 nu 에 AE-ID 를 준다.

        전에는 플래그 하나가 둘 다 정했다. 그래서 원격 때문에 pch 를 켜면
        poa 를 박는 코드가 direct 분기 안에만 있어서 같이 건너뛰었고,
        방화벽과 아무 상관없는 사업장 안의 텔레메트리 구독까지 죽었다.
        이제 로컬 설정은 원격이 무엇이든 그대로 돈다.
        """
        port = self.notification_port
        poa = f"http://{config.NOTIFY_HOST}:{port}"
        local_notify = f"http://127.0.0.1:{port}/notify"

        # ── 로컬: 사업장 MN-CSE ──────────────────────────
        local_ae = f"{self.mn_cse_name}/{self.ae_name}"
        if config.local_notify_mode() == "pch":
            self.local_pch = onem2m_pch.setup(
                local_ae, local_notify, label=f"{self.ae_name} / 로컬")
            self.local_notify_target = (
                self.local_pch.origin if self.local_pch else None)
        else:
            # rr=true 만 켜고 poa 를 비워 두면 CSE 가 알림 대상을 검증하지
            # 못해 구독이 거부된다("Verification request failed for: ...").
            # MN-CSE 는 같은 머신이지만 컨테이너 안이라 127.0.0.1 로는
            # 우리에게 못 온다. 호스트 주소를 알려줘야 한다.
            om2m.set_point_of_access(local_ae, [poa], reachable=True)
            self.local_notify_target = f"{poa}/notify"

        # ── 원격: 클라우드 IN-CSE ────────────────────────
        if config.remote_notify_mode() == "pch":
            self.pch_receiver = onem2m_pch.setup(
                om2m.in_path(self.ae_name), local_notify,
                label=f"{self.ae_name} / 원격")
            self.remote_notify_target = (
                self.pch_receiver.origin if self.pch_receiver else None)
            if not self.pch_receiver:
                print(f"    ⚠ {self.ae_name} 은 IN-CSE 로부터 알림을 받을 수 "
                      f"없다. 라운드 명령 구독이 실패할 것이다.")
        else:
            self.remote_notify_target = f"{poa}/notify"

    def _is_stale_command(self, command: dict) -> bool:
        """
        이 프로세스가 뜨기 전에 발행된 명령인가.

        cnt-fl-control 은 실행 사이에 비워지지 않는다. MN-AE 가 구독하고
        있어서 컨테이너를 지웠다 다시 만들 수 없기 때문이다. 그래서 지난
        실행의 FL_TRAINING round=1 이 그대로 남아 있고, 이번 실행의 것과
        내용이 똑같아 보인다. 발행 시각으로 가른다.
        """
        try:
            ts = float(command.get("timestamp", 0))
        except (TypeError, ValueError):
            return False
        if ts <= 0 or ts >= self.started_at:
            return False
        print(f"  ⓘ {self.node_name}: 지난 실행의 명령 무시 "
              f"(발행 {time.strftime('%H:%M:%S', time.localtime(ts))}, "
              f"기동 {time.strftime('%H:%M:%S', time.localtime(self.started_at))})")
        return True

    def _is_expected_command(self, command: dict | None, round_num: int) -> bool:
        if not isinstance(command, dict):
            return False

        state = str(command.get("jobState", ""))

        try:
            current_round = int(command.get("currentRound", command.get("round", 0)))
        except (TypeError, ValueError):
            current_round = 0

        # 핵심 수정:
        # 새 실행에서 MN-AE가 Round 1을 기다리는 중인데,
        # cnt-fl-control/la에 이전 실행의 FL_COMPLETED가 남아 있으면
        # 그걸 새 실행 완료로 착각하고 종료하는 문제가 생김.
        #
        # 따라서 Round 1 대기 중에는 FL_COMPLETED를 expected command로 인정하지 않음.
        # Round 1은 반드시 FL_TRAINING/currentRound=1로 시작해야 함.
        if state == "FL_COMPLETED":
            if round_num == 1:
                print(
                    f"  ⚠ {self.node_name}: ignore stale FL_COMPLETED "
                    f"while waiting Round 1"
                )
                return False

            # Round 2 이후에는 실제 완료 신호일 수 있으므로 허용.
            # 단, round 정보가 이상하면 무시함.
            return current_round >= round_num

        return (
            state == "FL_TRAINING"
            and current_round == round_num
        )

    def _wait_for_round_command(self, round_num: int):
        """
        NOTIFY를 주 경로로 사용함.
        최초 timeout 때 cnt-fl-control/la를 한 번만 조회하고,
        그 이후에는 다시 polling하지 않고 NOTIFY만 기다림.
        """
        fallback_checked = False

        while True:
            with self.job_lock:
                command = self.latest_job_command
                source = self.last_command_source

            if self._is_expected_command(command, round_num):
                return command, source or "NOTIFY"

            wait_timeout = (
                None
                if fallback_checked
                else self.command_wait_timeout_sec
            )

            notified = self.job_event.wait(timeout=wait_timeout)
            self.job_event.clear()

            if notified:
                continue

            # 최초 timeout에서만 단발성 fallback RETRIEVE 수행
            fallback_checked = True
            print(
                f"  ⚠ {self.node_name}: command NOTIFY timeout; "
                "check cnt-fl-control/la once"
            )

            fallback = self._get_current_job_state()
            if self._is_expected_command(fallback, round_num):
                with self.job_lock:
                    self.latest_job_command = fallback
                    self.last_command_source = "fallback RETRIEVE"
                return fallback, "fallback RETRIEVE"

            print(
                f"  ⚠ {self.node_name}: no matching command in fallback; "
                "continue waiting for NOTIFY"
            )

    # ---------------------------
    # oneM2M ops
    # ---------------------------
    def _subscribe_to_telemetry(self) -> int:
        """
        이 사업장 로봇들의 telemetry 컨테이너에 구독을 건다.

        설계 문서 9.8.5의 sub-sensor-data-X 에 해당한다. 로봇 AE(ADN-AE)는
        Isaac Sim 이 만들고, 우리는 그 아래 컨테이너에 구독만 건다.

        전 구간이 사업장 안이다. 로봇 -> MN-CSE -> MN-AE 모두 같은 네트워크라
        캠퍼스 방화벽과 무관하게 직접 알림이 닿는다. 실제로 CSE 파드가
        호스트의 5001 포트까지 도달하는 것을 확인했다.

        알림은 "새 데이터가 들어왔다"는 신호로만 쓴다. 내용은 라운드 수집
        때 cra 증분 조회로 한 번에 읽는다. 그래야 윈도우 하나당 왕복이
        생기지 않는다.

        반환: 성공한 구독 수
        """
        if not getattr(config, "FL_SENSOR_FROM_CSE", True):
            return 0

        # 텔레메트리는 사업장 안에서 오는 알림이다. 원격 경로와 무관하다.
        notify_url = self.local_notify_target
        if not notify_url:
            print(f"    ⚠ 로컬 알림 경로가 없다 — 구독을 걸지 않는다.")
            return 0

        ok = 0
        print(f"\n  [{self.node_name}] 로봇 텔레메트리 구독")
        for robot, tel_path in zip(self.robot_names, self.telemetry_paths):
            if not om2m.resource_exists(tel_path):
                print(f"    ⚠ {robot}: telemetry 없음 — 건너뜀")
                continue

            sub_name = f"sub-sensor-data-{self.node_name}"
            om2m.delete_subscription(tel_path, sub_name)
            time.sleep(0.1)

            res = om2m.create_subscription(
                parent_path=tel_path,
                subscription_name=sub_name,
                notification_uri=notify_url,
                event_types=[3],
                use_nct=1,
            )
            if not res:
                print(f"    ✗ {robot}: 구독 실패")
                continue

            ri = (res.get("m2m:sub") or {}).get("ri")
            if ri:
                self.telemetry_subs.add(ri)
            ok += 1
            print(f"    ✓ {robot}")

        if ok == 0:
            print(f"    ⚠ 구독이 하나도 안 걸렸다. 수집은 주기 조회로 돈다.")
        return ok

    def _subscribe_to_fl_control(self):
        """라운드 명령 구독. 대상은 클라우드 IN-CSE 라 원격 경로를 쓴다."""
        if not self.remote_notify_target:
            print(f"  ✗ {self.node_name}: IN-CSE 로부터 알림을 받을 경로가 "
                  f"없어 구독을 만들지 않는다.")
            return None

        sub_name = f"sub_{self.node_name}"
        om2m.delete_subscription(self.fl_control_path, sub_name)
        time.sleep(0.2)
        res = om2m.create_subscription(
            parent_path=self.fl_control_path,
            subscription_name=sub_name,
            notification_uri=self.remote_notify_target,
            event_types=[3],
            use_nct=None,
        )
        if res:
            print(f"  ✓ {self.node_name} subscribed to cnt-fl-control (net=3)")
        return res

    def _ensure_local_model_container(self):
        # config.CSE_NAME 은 import 시점의 mn1 CSE 라 mn2/mn3 가 쓰면
        # 남의 CSE 에 만들려 해서 404 가 난다. 자기 사업장 CSE 를 쓴다.
        parent = f"{self.mn_cse_name}/{self.ae_name}"
        res = om2m.create_container(
            parent_path=parent, container_name=self.local_model_cnt, mni=20
        )
        if res:
            print(f"  ✓ local cache container ready: {self.local_model_path}")

    def _get_current_job_state(self):
        cin = om2m.get_latest_content_instance(self.fl_control_path)
        if cin and "m2m:cin" in cin:
            con = cin["m2m:cin"]["con"]
            return json.loads(con) if isinstance(con, str) else con
        return None

    # ---------------------------
    # Sensor data load (AE 버전)
    # ---------------------------
    def _wait_for_robots(self) -> bool:
        """
        Isaac Sim이 이 사업장의 로봇 AE와 telemetry를 올릴 때까지 기다린다.

        샌드박스가 뜬 직후에는 MN-CSE만 있고 로봇 AE는 없다. Isaac이 등록한다.
        그래서 FL을 바로 시작하면 읽을 것이 없다. 라운드 안에서 기다리는 것과
        별개로, 기동 시점에 한 번 확실히 기다린다.

        기대 대수를 다 못 채워도 최소 대수만 준비되면 진행한다.
        로봇 한 대가 빠졌다고 사업장 전체를 멈출 이유는 없고, 뒤늦게 올라온
        로봇은 다음 라운드의 수집에서 자연히 포함된다.
        """
        timeout = float(os.getenv("FL_ROBOT_WAIT_SEC", "300"))
        min_robots = int(os.getenv("FL_MIN_ROBOTS_PER_SITE", "1"))
        poll = float(os.getenv("FL_ROBOT_POLL_SEC", "5"))

        site = config.SITE_OF_NODE.get(self.node_name, self.node_name)
        print(f"\n  [{self.node_name}/{site}] Isaac Sim 대기 "
              f"(최대 {timeout:.0f}s, 최소 {min_robots}대)")

        deadline = time.time() + timeout
        next_log = 0.0

        while True:
            ready = [
                rb for rb, path in zip(self.robot_names, self.telemetry_paths)
                if om2m.resource_exists(path)
            ]
            now = time.time()

            if len(ready) >= min_robots:
                print(f"    ✓ {len(ready)}/{len(self.robot_names)}대 준비됨 "
                      f"({', '.join(ready)})")
                missing = [r for r in self.robot_names if r not in ready]
                if missing:
                    print(f"    ⓘ 아직 없는 로봇: {', '.join(missing)} "
                          f"(나중에 올라오면 그 라운드부터 포함된다)")
                return True

            if now >= deadline:
                print(f"    ✗ {timeout:.0f}s 동안 로봇이 올라오지 않았다")
                print(f"      기대: {', '.join(self.robot_names)}")
                print(f"      확인: Isaac Sim이 {self.mn_cse_name} 에 "
                      f"등록 중인지")
                return False

            if now >= next_log:
                print(f"    ⏳ {len(ready)}/{min_robots}대 "
                      f"(남은 {deadline - now:.0f}s)")
                next_log = now + 10.0

            time.sleep(poll)

    def _seed_telemetry_cursor(self) -> None:
        """
        이미 쌓여 있는 텔레메트리를 건너뛰고, FL 시작 후에 들어온 것부터 읽는다.

        [왜]
        커서가 비어 있으면 fetch 가 "최근 N개"를 긁어 온다. 그래서 Round 1
        만 Isaac Sim 이 그때까지 올려둔 걸 통째로 가져갔다. 로그에서 mn1 이
        Round 1 에 341개를 먹고 Round 2 에는 37개밖에 안 남았던 것이 이것
        때문이다. 라운드마다 조건이 같아야 비교가 되므로, Round 1 도 다른
        라운드처럼 FL 이 시작된 뒤에 쌓인 것만 받아야 한다.

        각 로봇 컨테이너의 지금 마지막 ct 를 커서로 놓는다. 비어 있는
        컨테이너는 커서를 두지 않는다. 어차피 거기는 지금부터가 처음이다.
        """
        seeded, empty = 0, 0

        for path in self.telemetry_paths:
            try:
                cin = om2m.get_latest_content_instance(path)
            except Exception as e:
                print(f"    ⚠ {path} 마지막 ct 조회 실패(건너뜀): {e}")
                continue

            ct = ((cin or {}).get("m2m:cin") or {}).get("ct")
            if ct:
                self.telemetry_cursor[path] = ct
                seeded += 1
            else:
                empty += 1

        print(f"  ⏱ 쌓여 있던 텔레메트리는 건너뛴다 "
              f"(커서 설정 {seeded}개 / 비어 있음 {empty}개). "
              f"지금부터 쌓이는 것만 학습한다.")

    def _load_sensor_data_for_round(self, round_num: int,
                                    max_attempts: int = 3):
        """
        이 라운드에 쓸 데이터를 만든다. 쓸 만해질 때까지 기다린다.

        [왜 기다리나]
        예전에는 정상 주행 구간이 없으면 이 라운드를 통째로 건너뛰었다.
        건너뛴 노드는 아무것도 올리지 않고 다음 라운드로 가버리는데,
        IN-AE 는 그 노드의 업로드를 기다리느라 수집 타임아웃(600초)을
        통째로 버린다. 로그에서 mn1 이 Round 2 를 빠지자 IN-AE 가 (1/2)
        에서 10분을 멈춰 있던 것이 이 때문이다.

        빠질 일이 아니다. 로봇은 계속 돌고 있으니 조금 더 기다리면 정상
        주행 구간이 생긴다. 다른 노드와 같은 라운드에 참여하는 것이 맞다.

        [예산]
        IN-AE 의 수집 타임아웃보다 먼저 끝나야 의미가 있다. 기본값은
        FL_COLLECTION_NOTIFY_TIMEOUT 에서 60초를 뺀 값이고,
        FL_SENSOR_ROUND_MAX_WAIT_SEC 로 따로 줄 수 있다.
        그 안에 끝내 못 만들면 그때는 어쩔 수 없이 SKIP_ROUND 다.
        """
        in_timeout = float(os.getenv("FL_COLLECTION_NOTIFY_TIMEOUT", "600"))
        budget = float(os.getenv(
            "FL_SENSOR_ROUND_MAX_WAIT_SEC",
            str(max(60.0, in_timeout - 60.0))))
        poll = float(os.getenv("FL_SENSOR_ROUND_POLL_SEC", "2.0"))

        deadline = time.time() + budget
        attempt = 0
        next_log = 0.0

        while True:
            attempt += 1
            data = self._collect_round_data_once(
                round_num, hard_deadline=deadline)

            if data is None:
                # 텔레메트리가 아예 비었다. 기다려서 될 일이 아니다.
                return None

            if data is not SKIP_ROUND:
                # 이 라운드에 쓸 데이터가 정해졌다. 버퍼를 비운다.
                self._round_buffer = None
                return data

            left = deadline - time.time()
            if left <= 0:
                print(f"    ✗ {budget:.0f}초 동안 학습할 수 있는 구간이 "
                      f"생기지 않았다 (시도 {attempt}회).")
                print(f"      로봇이 주행 중인지 확인하라. "
                      f"이번 라운드는 빠진다.")
                return SKIP_ROUND

            now = time.time()
            if now >= next_log:
                print(f"    ⏳ 쓸 수 있는 구간이 없다. 라운드를 빠지지 않고 "
                      f"기다린다 (시도 {attempt}회, 남은 {left:.0f}s)")
                next_log = now + 15.0

            # 새 텔레메트리가 올 때까지 기다렸다 다시 본다.
            # 커서를 안 밀었으므로 다음 시도에는 지금 것 + 새로 온 것을
            # 같이 받는다. 기다릴수록 양이 늘어난다.
            if self.telemetry_subs:
                self.data_event.wait(timeout=poll)
                self.data_event.clear()
            else:
                time.sleep(poll)

    def _collect_round_data_once(self, round_num: int,
                                 hard_deadline: float = None):
        """수집 1회 시도. 쓸 데이터를 못 만들면 SKIP_ROUND 를 준다."""
        """
        이번 라운드에 학습할 윈도우를 모은다.

        예전 구조에서는 Isaac Sim이 cnt-sensor-data에 "라운드 N 데이터 준비됨"
        메타데이터 CIN을 올리고, MN-AE가 그 round 값이 맞을 때까지 기다렸다.
        실시간 스트림에는 그런 경계가 없다. 로봇은 계속 주행하고 있고,
        이번 라운드에는 방금까지 들어온 구간으로 학습하는 것이 자연스럽다.

        그래서 이렇게 바뀐다.
          1) 라운드 메타데이터를 기다리지 않는다. 최근 윈도우를 긁는다.
          2) 컨테이너가 로봇 대수만큼 있으므로 사업장 단위로 합친다.
          3) 윈도우가 너무 적으면 최소 개수가 모일 때까지만 잠깐 기다린다.

        FL_SENSOR_FROM_CSE=0 이면 로컬 pkl로 되돌아간다(오프라인 검증용).
        """
        site = config.SITE_OF_NODE.get(self.node_name, self.node_name)
        print(f"\n  [{self.node_name}/{site}] round_{round_num} 데이터 수집 ...")

        if not getattr(config, "FL_SENSOR_FROM_CSE", True):
            return self._load_sensor_data_from_pkl(round_num)

        # 라운드마다 사업장당 이만큼 모일 때까지 기다린다.
        # 상한(site_cap)과 같은 값이어야 라운드마다 정확히 같은 양이 된다.
        want_min   = int(os.getenv("FL_SENSOR_MIN_WINDOWS", "50"))

        # 사업장 단위 상한. 로봇 수가 사업장마다 다르므로(A 4대, B 2대)
        # 로봇당 상한만 두면 사업장 총량이 2배 차이 나고, FedAvg가 샘플 수로
        # 가중하기 때문에 로봇이 많은 사업장이 더 큰 영향력을 갖는다.
        # 사업장별 기여를 맞추려면 여기서 사업장 단위로 자른다.
        # 0으로 두면 자르지 않는다(예전 동작).
        site_cap = int(os.getenv("FL_SENSOR_MAX_WINDOWS_PER_SITE", "50"))

        # 로봇별 조회 상한은 사업장 상한에서 역산한다. 넉넉히 받아 두고
        # 합친 뒤에 고르게 줄이는 편이, 로봇 하나가 쉬고 있을 때도
        # 사업장 총량을 채울 수 있어 낫다.
        n_robots = max(1, len(self.telemetry_paths))
        default_per_robot = (max(50, site_cap // n_robots * 2)
                             if site_cap > 0 else 300)
        per_robot = int(os.getenv("FL_SENSOR_CIN_LIMIT_PER_ROBOT",
                                  str(default_per_robot)))
        # 하한이 찰 때까지 기다린다. 실제 상한은 바깥에서 넘겨주는
        # hard_deadline(라운드 예산)이므로 여기 기본값은 그보다 크게 둔다.
        # 예전 기본 90초는 50개가 쌓이기 전에 깨어나 모자란 채로 진행했다.
        wait_total = float(os.getenv("FL_SENSOR_ROUND_WAIT_SEC", "3600"))
        poll       = float(os.getenv("FL_SENSOR_ROUND_POLL_SEC", "2.0"))

        # 라운드마다 "최근 N개"를 다시 긁으면 간격이 짧을 때 같은 데이터를
        # 반복 학습하게 된다. 읽은 지점(ct)을 기억해 그 뒤만 받는다.
        # FL_SENSOR_INCREMENTAL=0 으로 끄면 예전처럼 최근 N개를 받는다.
        incremental = os.getenv("FL_SENSOR_INCREMENTAL", "1") != "0"

        deadline = time.time() + wait_total
        if hard_deadline is not None:
            deadline = min(deadline, hard_deadline)
        next_log = 0.0

        # 라운드가 바뀌면 버퍼를 비운다. 같은 라운드 안에서 재시도할 때는
        # 그대로 두어, 이미 챙긴 윈도우를 다시 받지 않아도 되게 한다.
        if self._round_buffer_round != round_num:
            self._round_buffer = None
            self._round_buffer_round = round_num

        n_fresh = 0        # 직전 조회에서 새로 받은 수
        n_fetched = 0      # 이 라운드에 조회로 받은 누계
        n_polls = 0

        while True:
            if incremental:
                fresh, cursors = amr_ingest.fetch_site_windows_since(
                    self.telemetry_paths, self.telemetry_cursor, per_robot)
                n_fresh = len(fresh["signals"])
                n_fetched += n_fresh
                n_polls += 1

                # 받은 즉시 커서를 민다. 안 밀면 다음 조회가 같은 구간을
                # 다시 받는데, telemetry 컨테이너는 mni 때문에 그 사이
                # 오래된 CIN 을 지운다. 그래서 총량이 로봇 수 × mni 에서
                # 멈춘다. 지워지기 전에 챙겨 버퍼에 쌓는 것이 맞다.
                self.telemetry_cursor = cursors
                self._round_buffer = amr_ingest.merge_windows(
                    self._round_buffer, fresh)
                got = self._round_buffer
            else:
                got = amr_ingest.fetch_site_windows(
                    self.telemetry_paths, per_robot)
            n = len(got["signals"])

            if n >= want_min:
                break

            now = time.time()
            if now >= deadline:
                if n == 0 and incremental and self.telemetry_cursor:
                    print(f"    ⊘ 지난 라운드 이후 새 윈도우가 없다.")
                    print(f"      로봇이 주행 중인지, 라운드 간격이 너무 "
                          f"짧지 않은지 확인하라.")
                    return SKIP_ROUND   # 바깥에서 다시 기다린다
                if n == 0:
                    print(f"    ✗ {site} 로봇 텔레메트리가 비어 있다.")
                    print(f"      확인: {len(self.telemetry_paths)}개 컨테이너 "
                          f"({', '.join(self.robot_names)})")
                    print("      Isaac Sim이 이 MN-CSE에 등록·전송 중인지 보라.")
                    return None
                print(f"    ⚠ 윈도우 {n}개로 진행한다 (최소 {want_min}개 기대)")
                break

            if now >= next_log:
                via = "구독" if self.telemetry_subs else "주기 조회"
                # 새로 받은 수(+N)를 같이 찍는다. 계속 +0 이면 CSE 가
                # 커서 이후 CIN 을 안 주는 것이고, +N 인데 왼쪽 숫자가
                # 안 늘면 쌓는 쪽이 잘못된 것이다. 로그만 보고 가려진다.
                print(f"    ⏳ 윈도우 {n}/{want_min} 수집 중 "
                      f"(직전 +{n_fresh}, 조회누계 {n_fetched}, "
                      f"{n_polls}회, {via}, 남은 {deadline - now:.0f}s)")
                next_log = now + 5.0

            # 구독이 걸려 있으면 알림이 깨울 때까지 기다린다.
            # 주기적으로 CSE를 찌르지 않아도 새 데이터가 오는 순간 반응한다.
            # 알림이 유실돼도 poll 초마다 한 번은 깨어나므로 멈추지 않는다.
            if self.telemetry_subs:
                self.data_event.wait(timeout=poll)
                self.data_event.clear()
            else:
                time.sleep(poll)

        # 커서는 위 루프에서 받을 때마다 이미 밀었다.
        # 받은 윈도우가 쓸 수 없는 것이어도 버퍼에 남아 있으므로,
        # 재시도할 때 다시 받을 필요가 없다.

        # 사업장 단위로 고르게 줄인다.
        if site_cap > 0 and len(got["signals"]) > site_cap:
            before = len(got["signals"])
            got = amr_ingest.balance_across_robots(got, site_cap)
            print(f"    ⓘ 사업장 상한 적용: {before} -> "
                  f"{len(got['signals'])}개 (로봇별 균등)")

        if self.telemetry_subs:
            print(f"    ⓘ 이번 수집까지 받은 알림 {self.data_notifications}건 "
                  f"(구독 {len(self.telemetry_subs)}개)")

        robots_seen = sorted(set(got["robot"].tolist()))
        print(f"    ✓ 윈도우 {len(got['signals'])}개 / 로봇 {len(robots_seen)}대 "
              f"(CIN {got['n_raw']}개"
              f"{', 증분' if incremental else ', 최근 N개'})")
        if len(robots_seen) < len(self.robot_names):
            missing = [r for r in self.robot_names if r not in robots_seen]
            print(f"    ⚠ 데이터가 없는 로봇: {', '.join(missing)}")

        return self._build_round_data_from_cse(got)

    def _load_sensor_data_from_pkl(self, round_num: int):
        """
        오프라인 폴백. prepare_data_amr.py가 만든 pkl을 라운드 수로 등분해
        이번 라운드 몫만 쓴다. CSE 없이 학습 로직만 돌려볼 때 쓴다.
        """
        path = config.DATA_PATHS.get(self.node_name)
        if not path or not os.path.exists(path):
            print(f"    ✗ pkl 없음: {path}")
            return None

        with open(path, "rb") as f:
            data_dict = pickle.load(f)

        train_sigs = data_dict.get("train_signals", np.array([]))
        total = len(train_sigs)
        if total == 0:
            print("    ✗ pkl에 train_signals가 없다")
            return None

        if config.GLOBAL_ROUNDS > 1:
            per = max(1, total // config.GLOBAL_ROUNDS)
            start = (round_num - 1) * per
            end = total if round_num >= config.GLOBAL_ROUNDS else start + per
        else:
            start, end = 0, total

        start = min(start, total - 1)
        end = max(end, start + 1)

        data_dict = dict(data_dict)
        data_dict["train_signals"] = train_sigs[start:end]
        if data_dict.get("train_cond") is not None:
            data_dict["train_cond"] = data_dict["train_cond"][start:end]
        data_dict["active_round"] = int(round_num)
        data_dict["round_start_idx"] = int(start)
        data_dict["round_end_idx"] = int(end)
        data_dict["sensor_meta"] = {"source": "pkl", "round": int(round_num)}

        print(f"    ✓ pkl 폴백 [{start}:{end}] "
              f"({end - start}/{total} 윈도우)")
        return data_dict


    def _build_round_data_from_cse(self, got: dict) -> dict:
        """
        CSE에서 받은 윈도우를 학습용 dict로 만든다.

        정지 구간은 결함이 드러날 여지가 거의 없어 뺀다.
        정규화 통계는 이 노드가 보관한 것을 쓰고, 없으면 받은 데이터로 잡는다.
        사업장·로봇마다 정상의 모습이 달라 공용 통계를 쓰면 정상도 이상으로 보인다.
        """
        import amr_schema as sch

        X, C = got["signals"], got["cond"]
        mv = sch.moving_mask(X)
        if mv.any():
            X, C = X[mv], C[mv]
            robot = got["robot"][mv]
            names = got["cond_name"][mv]
        else:
            robot, names = got["robot"], got["cond_name"]

        Xn = amr_ingest.normalize_with(X, self.norm_mean, self.norm_std)
        if self.norm_mean is None and len(Xn):
            # 첫 수집이면 이 데이터로 통계를 잡고 이후 라운드에서 재사용한다.
            self.norm_mean = X.mean(axis=(0, 2), keepdims=True)
            self.norm_std = X.std(axis=(0, 2), keepdims=True) + 1e-8

        # 학습은 정상 주행만 쓴다.
        #
        # 예전에는 정상이 하나도 없으면 전부 학습 대상으로 돌렸는데, 그건
        # 이상 데이터로 정상 패턴을 배우는 꼴이라 목적과 정면으로 충돌한다.
        # 더 나쁜 것은 그 결과가 FedAvg를 타고 다른 사업장까지 번진다는 점이다.
        # 그래서 지금은 되돌리지 않고 이 라운드를 건너뛴다.
        #
        # 제외 nav(중단된 주행 등)도 오프라인 전처리와 같은 기준으로 뺀다.
        # 온라인과 오프라인이 다른 학습 집합을 쓰면 비교가 안 된다.
        # 판정에 쓸 윈도우를 남겨 둔다.
        #
        # 학습 대상(정상)만 추리기 전의 '이동 중인 윈도우 전부'가 대상이다.
        # 이상을 찾는 것이 목적인데 정상만 보면 찾을 것이 없다.
        cts = got.get("ct")
        if cts is not None and len(cts) == len(mv):
            cts = cts[mv] if mv.any() else cts

        # 판정 CIN 에 "바닥 이상 위치"를 싣기 위해 윈도우별 좌표도 남긴다.
        # pose_of 가 윈도우 안 샘플의 평균 (x, y) 로 만들어 둔 값이다.
        # 정지 구간을 걸러낸 mv 마스크를 다른 배열과 똑같이 적용해야
        # 인덱스가 어긋나 엉뚱한 로봇의 좌표가 붙지 않는다.
        poses = got.get("pose")
        if poses is not None and len(poses) == len(mv):
            poses = poses[mv] if mv.any() else poses

        self.last_windows = {
            "X": Xn, "C": C, "robot": robot, "ct": cts, "pose": poses,
            # 대시보드가 이상 구간을 표시할 때 쓴다.
            # 이상 주입을 끈 실험에서는 전부 0 이다.
            "labels": (names != "normal").astype(np.int64),
        }

        navs = got.get("nav")
        if navs is not None and len(navs) == len(mv):
            navs = navs[mv] if mv.any() else navs
        is_normal = names == "normal"
        if navs is not None and len(navs) == len(is_normal):
            is_normal = is_normal & ~np.isin(navs, list(sch.EXCLUDE_NAV_TRAIN))

        n_normal = int(is_normal.sum())
        print(f"      이동 {len(Xn)}개 중 학습 대상 {n_normal}개")

        if n_normal == 0:
            print("      ✗ 정상 주행 구간이 없다. 이 라운드는 학습하지 않는다.")
            print("        (이상 구간으로 정상 패턴을 학습할 수는 없다)")
            return SKIP_ROUND

        return {
            "train_signals": Xn[is_normal],
            "train_cond": C[is_normal],
            "val_signals": Xn,
            "val_cond": C,
            "val_labels": (names != "normal").astype(np.int64),
            "from_cse": True,
            "robots": sorted(set(robot.tolist())),
        }

    def _build_ae_node_from_data_dict(self, data_dict: dict) -> AEEdgeNode | None:
        """
        PKL dict → AEEdgeNode
        PKL 형식 (prepare_data_amr.py 출력):
          train_signals, val_signals, val_labels, norm_mean, norm_std, ...
        """
        train_sigs = data_dict.get("train_signals")
        val_sigs   = data_dict.get("val_signals")
        val_labels = data_dict.get("val_labels")

        if train_sigs is None or len(train_sigs) == 0:
            print(f"    ✗ train_signals 없음")
            return None

        if val_sigs is None:
            val_sigs   = np.empty((0, train_sigs.shape[1]), dtype=np.float32)
            val_labels = np.empty((0,), dtype=np.int64)

        # 이상 노드 시뮬레이션 (테스트용)
        if self.inject_anomaly:
            noise_scale = 2.0  # 정규화된 신호 기준이므로 작게
            print(f"  ⚠ [ANOMALY] 센서 데이터 노이즈 주입 (scale={noise_scale})")
            train_sigs = train_sigs + np.random.randn(*train_sigs.shape).astype(
                np.float32
            ) * noise_scale

        train_cfg = getattr(config, "TRAIN_CFG", config.TrainConfig())
        ae_cfg    = getattr(config, "AE_CFG",    config.AEConfig())

        edge_node = AEEdgeNode(
            node_id       = self.node_id,
            train_signals = train_sigs,
            val_signals   = val_sigs,
            val_labels    = val_labels,
            ae_cfg        = ae_cfg,
            train_cfg     = train_cfg,
            device        = "cpu",
            train_cond    = data_dict.get("train_cond"),
            val_cond      = data_dict.get("val_cond"),
            train_run_id  = data_dict.get("train_run_id"),
            val_run_id    = data_dict.get("val_run_id"),
        )
        return edge_node

    # ---------------------------
    # Local cache metadata
    # ---------------------------
    def _read_local_cache_meta(self):
        if not os.path.exists(self.local_cache_meta_file):
            return None
        try:
            with open(self.local_cache_meta_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _write_local_cache_meta(self, cached_round: int, local_path: str, src_uri: str):
        payload = {
            "type":       "GLOBAL_MODEL_CACHE",
            "round":      cached_round,
            "model_path": local_path,
            "sha256":     (
                self._sha256_file(local_path) if os.path.exists(local_path) else ""
            ),
            "src_uri":    src_uri,
            "timestamp":  time.time(),
        }
        try:
            with open(self.local_cache_meta_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            return payload
        except Exception as e:
            print(f"    ⚠ local cache meta write failed: {e}")
            return None

    # ---------------------------
    # Edge cache
    # ---------------------------
    def _read_cache_meta_la(self):
        local = self._read_local_cache_meta()
        if local:
            return local
        cin = om2m.get_latest_content_instance(self.local_model_path)
        if not cin or "m2m:cin" not in cin:
            return None
        con = cin["m2m:cin"]["con"]
        try:
            obj = json.loads(con) if isinstance(con, str) else con
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def _write_cache_meta(
        self,
        cached_round: int,
        local_path: str,
        src_uri: str
    ):
        payload = self._write_local_cache_meta(
            cached_round,
            local_path,
            src_uri
        )

        if not payload:
            return None

        payload["type"] = "global-cache"

        result = om2m.create_content_instance(
            self.local_model_path,
            payload
        )

        if result:
            print(
                f"    ✓ cached meta -> {self.local_model_path} "
                f"(round {cached_round})"
            )
        else:
            print(
                f"    ⚠ cache meta mirror failed "
                f"(local OK, round {cached_round})"
            )
        return payload

    def _try_load_global_from_cache(
        self,
        expected_round: int,
        expected_src_uri: str | None = None
    ):
        meta = self._read_cache_meta_la()
        if not meta:
            return None
        try:
            cached_round = int(meta.get("round", -1))
        except Exception:
            return None
        if cached_round != expected_round:
            return None

        cached_src_uri = str(meta.get("src_uri", ""))
        if expected_src_uri and cached_src_uri != str(expected_src_uri):
            print(
                "    ⚠ cache source URI differs from current command "
                "-> refresh cache"
            )
            return None

        path = meta.get("model_path")
        if not path or not os.path.exists(path):
            return None
        sha = meta.get("sha256", "")
        if sha:
            try:
                if sha != self._sha256_file(path):
                    print("    ⚠ cache sha256 mismatch -> ignore cache")
                    return None
            except Exception:
                return None
        return path

    def _remove_bad_cache(self, model_path: str):
        try:
            if model_path and os.path.exists(model_path):
                os.remove(model_path)
        except OSError:
            pass

        try:
            if os.path.exists(self.local_cache_meta_file):
                os.remove(self.local_cache_meta_file)
        except OSError:
            pass

    def _load_global_state_safely(self, model_path: str) -> bool:
        try:
            state_dict = torch.load(model_path, map_location="cpu")
            self.edge_node.set_state_dict(state_dict)
            return True
        except RuntimeError as exc:
            print(f"    ⚠ incompatible model checkpoint: {model_path}")
            print(f"    ⚠ {exc}")
            self._remove_bad_cache(model_path)
            return False
        except Exception as exc:
            print(f"    ⚠ failed to load model {model_path}: {exc}")
            return False

    def _get_global_model_meta_for_expected_round(
        self,
        expected_round: int,
        global_model_uri: str | None = None
    ):
        """
        글로벌 모델 contentInstance의 con 전체를 돌려준다.
        예전에는 con 안의 model_path만 꺼내 돌려줬지만, 이제는 con에 가중치
        자체(weights_b64)가 실려 오므로 호출측이 둘 중 하나를 고를 수 있어야 한다.
        """
        cin = None

        if global_model_uri:
            resource_path = str(global_model_uri)
            if resource_path.startswith(config.BASE_URL):
                resource_path = resource_path[len(config.BASE_URL):]
            resource_path = resource_path.lstrip("/")
            cin = om2m.get_resource(resource_path)

        # exact URI 조회가 실패한 경우에만 /la를 한 번 확인
        if not cin:
            cin = om2m.get_latest_content_instance(
                self.global_model_path
            )

        if not cin or "m2m:cin" not in cin:
            return None

        con = cin["m2m:cin"].get("con")
        try:
            model_data = (
                json.loads(con)
                if isinstance(con, str)
                else con
            )
        except (TypeError, json.JSONDecodeError):
            return None

        if not isinstance(model_data, dict):
            return None

        try:
            global_round = int(model_data.get("global_round", -1))
        except (TypeError, ValueError):
            return None

        if global_round != expected_round:
            print(
                f"    ⚠ global model round mismatch: "
                f"expected={expected_round}, got={global_round}"
            )
            return None

        return model_data

    def _pull_and_cache_global_model(
        self,
        expected_round: int,
        global_model_uri: str | None = None
    ):
        meta = self._get_global_model_meta_for_expected_round(
            expected_round,
            global_model_uri=global_model_uri
        )

        if not meta:
            return None

        dst_path = os.path.join(
            self.edge_cache_dir,
            f"global_round{expected_round}.pt"
        )

        # con에 가중치가 실려 왔으면 복원해서 캐시에 저장한다.
        # 이 경로만으로 동작하면 MN이 다른 머신에 있어도 FL이 성립한다.
        state_dict = None
        try:
            state_dict = model_codec.decode_state_dict(
                meta,
                fetch_chunk=self._make_chunk_fetcher(meta),
            )
        except Exception as e:
            print(
                f"    ⚠ con 가중치 복원 실패 "
                f"({type(e).__name__}: {e})"
            )

        if state_dict is not None:
            try:
                torch.save(state_dict, dst_path)
                print("    ↓ global weights from con(base64)")
            except Exception as e:
                print(f"    ⚠ 캐시 저장 실패: {e}")
                return None
        else:
            # 호환 모드: 같은 PC에 있을 때만 통하는 예전 방식
            src_path = meta.get("model_path")
            if not src_path or not os.path.exists(src_path):
                print(
                    "    ✗ con에 가중치가 없고 model_path도 접근 불가 "
                    "(MN이 IN과 다른 머신이면 FL_EMBED_WEIGHTS=1 필요)"
                )
                return None
            try:
                shutil.copyfile(src_path, dst_path)
                print("    ↓ global weights via model_path (호환 모드)")
            except Exception as e:
                print(f"    ⚠ copy failed: {e}")
                return None

        self._write_cache_meta(
            cached_round=expected_round,
            local_path=dst_path,
            src_uri=(
                global_model_uri
                or f"{self.global_model_path}/la"
            ),
        )

        return dst_path

    # ---------------------------
    # 가중치 조각 (CSE 페이로드 상한 대응)
    # ---------------------------
    def _ensure_weights_container(self, parent_path: str) -> str:
        """
        조각을 쌓을 자식 컨테이너를 보장한다.
        부모가 아니라 자식에 쌓는 이유: 부모 컨테이너에는 구독이 걸려 있어서
        조각 수십 개가 전부 알림을 발생시키면 수신측이 폭주한다.
        자식에 쌓으면 CREATE_OF_DIRECT_CHILD 조건에 걸리지 않는다.
        """
        om2m.create_container(
            parent_path,
            "cnt-weights",
            mni=1000,
            mbs=60_000_000,
        )
        return f"{parent_path}/cnt-weights"

    def _upload_weight_chunks(self, round_num: int, enc: dict):
        """
        가중치 base64를 CIN 상한에 맞게 조각내어 올리고, 매니페스트를 돌려준다.
        조각 하나라도 실패하면 None을 돌려 호출측이 폴백하도록 한다.
        """
        chunks, manifest = model_codec.split_for_cin(
            enc,
            getattr(config, "FL_MAX_CIN_BYTES", 48 * 1024),
        )

        cnt_path = self._ensure_weights_container(self.dropbox_cnt_path)

        for i, part in enumerate(chunks):
            res = om2m.create_content_instance(
                cnt_path,
                {"i": i, "d": part},
                resource_name=model_codec.chunk_resource_name(round_num, i),
            )
            if not res:
                print(
                    f"    ✗ 가중치 조각 {i + 1}/{len(chunks)} 업로드 실패"
                )
                return None

        manifest["weights_chunk_container"] = cnt_path
        manifest["weights_chunk_round"] = round_num
        return manifest

    def _make_chunk_fetcher(self, meta: dict):
        """
        매니페스트를 보고 조각을 이름으로 직접 RETRIEVE 하는 함수를 만든다.
        discovery를 쓰지 않으므로 조각 순서가 보장된다.
        """
        cnt_path = meta.get("weights_chunk_container")
        if not cnt_path:
            return None

        chunk_round = meta.get(
            "weights_chunk_round",
            meta.get("global_round", meta.get("round")),
        )

        def fetch(index: int):
            res = om2m.get_resource(
                f"{cnt_path}/"
                f"{model_codec.chunk_resource_name(chunk_round, index)}"
            )
            if not res:
                return None
            con = (res.get("m2m:cin") or {}).get("con")
            try:
                obj = json.loads(con) if isinstance(con, str) else con
            except (TypeError, ValueError):
                return None
            return (obj or {}).get("d")

        return fetch

    # ---------------------------
    # Drop-box upload
    # ---------------------------
    def _window_scores_block(self) -> dict:
        """
        이번 라운드 윈도우들의 복원오차와 임계값.

        대시보드는 원래 /tmp 의 PKL 파일을 읽어 직접 점수를 계산했다.
        데이터 출처가 CSE(Isaac Sim 텔레메트리)로 바뀌면서 그 파일이
        없어졌고, 노드 패널이 빈칸이 됐다. 이미 계산해 둔 점수를
        드롭박스 CIN 에 같이 실어 보내면 대시보드는 그리기만 하면 된다.

        판정(verdict)과 달리 FL_VERDICT 와 무관하게 항상 싣는다.
        점수 계산이 실패해도 업로드는 계속돼야 하므로 빈 dict 를 준다.
        """
        w = getattr(self, "last_windows", None)
        if not w or w.get("X") is None or len(w["X"]) == 0:
            return {}

        node = getattr(self, "edge_node", None)
        if node is None:
            return {}

        try:
            scores = node.compute_anomaly_scores(w["X"], w["C"])
        except Exception as e:
            print(f"    ⚠ 윈도우 점수 계산 실패(업로드는 계속): {e}")
            return {}

        thr = getattr(node, "threshold", None)
        labels = w.get("labels")
        robots = w.get("robot")

        block = {
            "window_scores": [round(float(v), 6) for v in scores],
            "threshold": (float(thr) if thr is not None else None),
        }
        if labels is not None and len(labels) == len(scores):
            block["window_labels"] = [int(v) for v in labels]
        if robots is not None and len(robots) == len(scores):
            block["window_robots"] = [str(v) for v in robots]
        return block

    def _publish_verdicts(self, round_num: int,
                          model_uri: str = None) -> int:
        """
        이번 라운드에 읽은 윈도우로 로봇별 이상 판정을 내고 CSE 에 올린다.

        학습이 끝난 뒤에 부른다. 임계값이 이번 라운드 모델 기준으로
        갱신돼 있어야 판정이 의미를 갖는다.

        판정이 실패해도 라운드는 계속된다. 연합학습 본체가 부수 기능 때문에
        멈추면 안 된다.
        """
        if not getattr(config, "VERDICT_ENABLED", True):
            return 0

        w = self.last_windows
        if not w or w["X"] is None or len(w["X"]) == 0:
            return 0

        node = self.edge_node
        threshold = getattr(node, "threshold", None)
        if threshold is None:
            return 0

        try:
            scores = node.compute_anomaly_scores(w["X"], w["C"])
            return verdict.publish(
                node_name=self.node_name,
                cnt_path=self.verdict_cnt_path,
                origin=self.analysis_ae_id,
                scores=scores,
                robots=w["robot"],
                threshold=threshold,
                round_num=round_num,
                cts=w["ct"],
                poses=w.get("pose"),
                model_uri=model_uri,
            )
        except Exception as e:
            print(f"  ⚠ {self.node_name}: 판정 실패 — {e}")
            return 0

    def _upload_inference_result(
        self,
        round_num,
        val_loss,
        val_auroc,
        train_loss: float = 0.0,
        num_samples: int = 0,
        local_only: bool = False,
    ):
        """
        FedAvg에 기여하지 않는 노드의 업로드.

        관측 전용(cold start)과 로컬 전용 둘 다 여기로 온다.
        어느 쪽이든 가중치와 model_path를 싣지 않으므로 집계 대상이 아니다.
        type을 fl-inference로 두어 IN-AE가 구분하고,
        대시보드는 지표만 받아 표시한다.
        """
        payload = {
            "type": "fl-inference",
            "node": self.node_name,
            "round": round_num,
            "cold_start": bool(self.is_cold_start),
            "local_only": bool(local_only or self.is_local_only),

            # 로컬 전용은 학습을 했으므로 실제 값이 들어간다.
            # 관측 전용은 0이다.
            "train_loss": float(train_loss),
            "num_samples": int(num_samples),

            "accuracy": val_auroc,
            "val_loss": val_loss,
            "val_auroc": val_auroc,
            "threshold": getattr(getattr(self, "edge_node", None), "threshold", None),
            "timestamp": time.time(),
        }

        # 가중치는 안 올리지만 대시보드 그래프는 떠야 한다.
        payload.update(self._window_scores_block())

        result = om2m.create_content_instance(
            self.dropbox_cnt_path,
            payload
        )

        if result:
            print(
                f"  ✓ inference result -> {self.dropbox_cnt_path} "
                f"(round {round_num}, 가중치 없음)"
            )

        return result

    def _upload_to_dropbox(
        self,
        round_num,
        train_loss,
        val_loss,
        val_auroc,
        num_samples,
        model_path
    ):
        payload = {
            "type": "fl-update",
            "node": self.node_name,
            "round": round_num,

            # 호환용 필드. 같은 PC에서 돌던 기존 방식을 위해 남겨두지만,
            # 집계측은 아래에서 동봉하는 weights_b64를 우선 사용한다.
            "model_path": model_path,
            "weights_uri": model_path,

            "accuracy": val_auroc,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auroc": val_auroc,
            "num_samples": num_samples,
            "timestamp": time.time(),
        }

        # 대시보드 그래프용 윈도우별 점수. 50 윈도우면 500 바이트 정도라
        # 가중치 크기 제한(FL_MAX_CIN_BYTES)에 영향을 주지 않는다.
        payload.update(self._window_scores_block())

        # 실제 가중치를 con 안에 실어 보낸다.
        # 이것이 있어야 MN을 다른 머신에 올려도 FL이 성립한다.
        if getattr(config, "FL_EMBED_WEIGHTS", False):
            try:
                enc = model_codec.encode_state_dict(
                    torch.load(model_path, map_location="cpu"),
                    fp16=getattr(config, "FL_WEIGHTS_FP16", True),
                )
                limit = getattr(config, "FL_MAX_CIN_BYTES", 48 * 1024)

                if enc["weights_bytes"] <= limit:
                    payload.update(enc)
                    print(
                        f"    ↑ weights in con: {model_codec.describe(enc)}"
                    )
                elif getattr(config, "FL_CHUNK_WEIGHTS", True):
                    manifest = self._upload_weight_chunks(round_num, enc)
                    if manifest:
                        payload.update(manifest)
                        print(
                            f"    ↑ weights in con: "
                            f"{model_codec.describe(manifest)} "
                            f"-> {manifest['weights_chunk_container']}"
                        )
                    else:
                        print(
                            "    ⚠ 조각 업로드 실패 "
                            "-> model_path 방식으로 대체"
                        )
                else:
                    print(
                        f"    ⚠ weights {enc['weights_bytes'] / 1024:.0f}KB "
                        f"> limit {limit / 1024:.0f}KB, 조각 전송 꺼짐 "
                        f"-> model_path 방식으로 대체"
                    )
            except Exception as exc:
                print(
                    f"    ⚠ weight embedding 실패 ({type(exc).__name__}: {exc}) "
                    f"-> model_path 방식으로 대체"
                )

        result = om2m.create_content_instance(
            self.dropbox_cnt_path,
            payload
        )

        if result:
            print(
                f"  ✓ uploaded -> {self.dropbox_cnt_path} "
                f"(round {round_num})"
            )

        return result

    # ---------------------------
    # Main loop
    # ---------------------------
    def run(self):
        print(f"\n=== {self.node_name.upper()} Trainer start (AE 버전) ===")
        print(f"  AE:           {self.ae_name}")
        print(f"  telemetry:    {len(self.telemetry_paths)}개 컨테이너 "
              f"({', '.join(self.robot_names)})")
        print(f"  local-model:  {self.local_model_path} (Edge cache)")
        print(f"  fl-control:   {self.fl_control_path}")
        print(f"  global-model: {self.global_model_path}")
        print(f"  drop-box:     {self.dropbox_cnt_path}")
        if self.is_cold_start:
            print(
                "  ❄ 관측 전용: 학습하지 않음 "
                "(글로벌 모델 수신 → 추론만 → 가중치 미전송)"
            )
        elif self.is_local_only:
            print(
                "  ◐ 로컬 전용: 자기 데이터로 학습하되 FedAvg에 기여하지 않음 "
                "(글로벌 모델 수신 → 로컬 보정 → 가중치 미전송)"
            )
        if self.inject_anomaly:
            print("  ⚠ ANOMALY MODE: 센서 데이터에 노이즈 주입 예정")

        self.start_server()

        if not self._subscribe_to_fl_control():
            print(f"  ✗ {self.node_name}: failed to subscribe to fl-control")
            return

        self._ensure_local_model_container()

        if not self._wait_for_robots():
            print(f"  ✗ {self.node_name}: 로봇 텔레메트리 없이는 학습할 수 없다")
            return

        # 로봇이 올라온 뒤에 구독을 건다. 없는 컨테이너에는 못 건다.
        self._subscribe_to_telemetry()

        # 이미 쌓여 있던 텔레메트리는 버리고 지금부터 받는다.
        # FL_SENSOR_SEED_CURSOR=0 이면 예전처럼 쌓인 것을 가져간다.
        if os.getenv("FL_SENSOR_SEED_CURSOR", "1") != "0":
            self._seed_telemetry_cursor()

        for round_num in range(1, config.GLOBAL_ROUNDS + 1):
            print("\n" + "─" * 50)
            print(f"[{self.node_name}] wait Round {round_num}")
            print("─" * 50)

            job_state, command_source = self._wait_for_round_command(
                round_num
            )

            state = str(job_state.get("jobState", ""))
            if state == "FL_COMPLETED":
                print(f"  ✓ {self.node_name} done: FL_COMPLETED")
                return

            print(
                f"  ✓ Round {round_num} command received "
                f"via {command_source} (jobState={state})"
            )

            if self.round_start_stagger_sec > 0:
                time.sleep(self.round_start_stagger_sec)

            # 데이터 로드 + AEEdgeNode 생성
            data_dict = self._load_sensor_data_for_round(round_num)

            if data_dict is SKIP_ROUND:
                # 여기까지 왔다는 건 라운드 예산(기본 540초) 내내 기다렸는데도
                # 학습할 수 있는 구간이 끝내 안 생겼다는 뜻이다. 로봇이 아예
                # 안 움직이는 상태라 코드가 더 할 수 있는 게 없다.
                # 평소에는 _load_sensor_data_for_round 가 데이터가 모일 때까지
                # 기다리므로 여기로 오지 않는다.
                print(
                    f"  ⊘ Round {round_num}: 학습 가능한 정상 구간이 없어 "
                    f"이번 라운드는 빠진다 ({self.node_name})"
                )
                continue

            if data_dict is None:
                print(
                    f"  ✗ failed to load data for round {round_num}; "
                    "stop MN-AE"
                )
                return

            self.edge_node = self._build_ae_node_from_data_dict(data_dict)
            if self.edge_node is None:
                print(
                    f"  ✗ failed to build AEEdgeNode for round {round_num}; "
                    "stop MN-AE"
                )
                return

            # round N은 global model round N-1을 사용
            expected_round = round_num - 1
            global_model_uri = job_state.get("globalModelUri")

            model_loaded = False
            cached_path = self._try_load_global_from_cache(
                expected_round,
                expected_src_uri=global_model_uri
            )

            if cached_path:
                model_loaded = self._load_global_state_safely(
                    cached_path
                )
                if model_loaded:
                    print(
                        f"    ✓ global model from EDGE cache "
                        f"(round {expected_round})"
                    )

            if not model_loaded:
                pulled_path = self._pull_and_cache_global_model(
                    expected_round,
                    global_model_uri=global_model_uri,
                )

                if not pulled_path:
                    print(
                        f"  ✗ global model unavailable "
                        f"for round {expected_round}; stop MN-AE"
                    )
                    return

                model_loaded = self._load_global_state_safely(
                    pulled_path
                )
                if not model_loaded:
                    print(
                        f"  ✗ incompatible global model "
                        f"for round {expected_round}; stop MN-AE"
                    )
                    return

                print(
                    f"    ✓ global model from IN-AE "
                    f"(round {expected_round})"
                )

            # DP 설정
            security_mode = job_state.get("securityMode", "DP")
            privacy_params = job_state.get("privacyParams", {})
            dp_epsilon = float(
                privacy_params.get(
                    "epsilon",
                    getattr(config, "DP_EPSILON", 8.0)
                )
            )
            dp_max_grad_norm = float(
                privacy_params.get(
                    "max_grad_norm",
                    getattr(config, "DP_MAX_GRAD_NORM", 1.5)
                )
            )
            dp_delta = float(
                privacy_params.get(
                    "delta",
                    getattr(config, "DP_DELTA", 1e-5)
                )
            )

            if self.is_cold_start:
                # 학습하지 않는다. 받은 글로벌 모델 그대로 자기 데이터를
                # 평가해서 임계값/AUROC만 구하고, 가중치 없이 올린다.
                print(
                    f"  [{self.node_name}] Round {round_num} "
                    f"COLD START — 학습 없음, 글로벌 모델로 추론만"
                )

                val_loss, val_auroc = self.edge_node._evaluate_val()

                print(
                    f"  ✓ inference only"
                    f"  val_loss={val_loss:.6f}"
                    f"  val_auroc={val_auroc:.4f}"
                )

                if self.upload_stagger_sec > 0:
                    time.sleep(self.upload_stagger_sec)

                if not self._upload_inference_result(
                    round_num=round_num,
                    val_loss=val_loss,
                    val_auroc=val_auroc,
                ):
                    print(
                        f"  ✗ inference result upload failed "
                        f"in round {round_num}; stop MN-AE"
                    )
                    return

                continue

            if security_mode == "DP":
                print(
                    f"  [{self.node_name}] Round {round_num} training..."
                    f" DP (ε={dp_epsilon}, C={dp_max_grad_norm})"
                )
            else:
                print(
                    f"  [{self.node_name}] Round {round_num} training..."
                    f" {security_mode} (DP 비활성)"
                )
                dp_epsilon = None

            train_loss, val_loss, val_auroc = self.edge_node.train_local(
                dp_epsilon=dp_epsilon,
                dp_delta=dp_delta,
                dp_max_grad_norm=dp_max_grad_norm,
            )

            os.makedirs(
                f"/tmp/fl_models/local/{self.node_name}",
                exist_ok=True
            )
            model_path = (
                f"/tmp/fl_models/local/{self.node_name}/"
                f"round{round_num}.pt"
            )
            torch.save(self.edge_node.get_state_dict(), model_path)

            print(
                f"  ✓ train_loss={train_loss:.6f}"
                f"  val_loss={val_loss:.6f}"
                f"  val_auroc={val_auroc:.4f}"
            )

            # 학습이 끝나 임계값이 갱신된 뒤에 판정한다.
            self._publish_verdicts(round_num)

            if self.upload_stagger_sec > 0:
                time.sleep(self.upload_stagger_sec)

            if self.is_local_only:
                # 학습은 했지만 가중치는 올리지 않는다.
                # 공통 모델을 받아 자기 데이터로 보정하는 사업장이다.
                # 중앙은 이 노드를 FedAvg와 집계 대기에서 제외한다.
                print(
                    f"  ◐ 로컬 전용: 가중치 미전송 "
                    f"(지표만 보고, 임계값={self.edge_node.threshold})"
                )
                upload_result = self._upload_inference_result(
                    round_num=round_num,
                    val_loss=val_loss,
                    val_auroc=val_auroc,
                    train_loss=train_loss,
                    num_samples=self.edge_node.num_train_samples,
                    local_only=True,
                )
            else:
                upload_result = self._upload_to_dropbox(
                    round_num=round_num,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_auroc=val_auroc,
                    num_samples=self.edge_node.num_train_samples,
                    model_path=model_path,
                )

            if not upload_result:
                print(
                    f"  ✗ local update upload failed in round {round_num}; "
                    "stop MN-AE"
                )
                return


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 mn_ae_standard.py <node_id> <port> [--anomaly]")
        print("Example: python3 mn_ae_standard.py 0 5001")
        print("Example: python3 mn_ae_standard.py 2 5003 --anomaly")
        raise SystemExit(1)

    node_id        = int(sys.argv[1])
    port           = int(sys.argv[2])
    inject_anomaly = "--anomaly" in sys.argv

    if inject_anomaly:
        print(f"⚠ [ANOMALY MODE] node_id={node_id}")

    MNAETrainer(node_id, port, inject_anomaly=inject_anomaly).run()