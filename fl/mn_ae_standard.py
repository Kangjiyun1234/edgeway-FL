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

                # 정상 경로:
                # round command CIN의 con을 NOTIFY body에서 읽음
                command = self._parse_cin_con_from_notification(sgn)

                # CSE가 rep를 보내지 않은 경우에만
                # NOTIFY를 계기로 최신 CIN을 한 번 조회
                if command is None:
                    command = self._get_current_job_state()

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

                    elif state in ("FL_TRAINING", "FL_COMPLETED"):
                        with self.job_lock:
                            self.latest_job_command = command
                            self.last_command_source = "NOTIFY"

                        print(
                            f"  [NOTIFY] {self.node_name} "
                            f"command received: "
                            f"{state}, round={current_round}"
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
    def _subscribe_to_fl_control(self):
        sub_name = f"sub_{self.node_name}"
        om2m.delete_subscription(self.fl_control_path, sub_name)
        time.sleep(0.2)
        res = om2m.create_subscription(
            parent_path=self.fl_control_path,
            subscription_name=sub_name,
            notification_uri=(
                f"http://{config.NOTIFY_HOST}:{self.notification_port}/notify"
            ),
            event_types=[3],
            use_nct=None,
        )
        if res:
            print(f"  ✓ {self.node_name} subscribed to cnt-fl-control (net=3)")
        return res

    def _ensure_local_model_container(self):
        parent = f"{config.CSE_NAME}/{self.ae_name}"
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

    def _load_sensor_data_for_round(self, round_num: int, max_attempts: int = 3):
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

        want_min   = int(os.getenv("FL_SENSOR_MIN_WINDOWS", "40"))

        # 사업장 단위 상한. 로봇 수가 사업장마다 다르므로(A 4대, B 2대)
        # 로봇당 상한만 두면 사업장 총량이 2배 차이 나고, FedAvg가 샘플 수로
        # 가중하기 때문에 로봇이 많은 사업장이 더 큰 영향력을 갖는다.
        # 사업장별 기여를 맞추려면 여기서 사업장 단위로 자른다.
        # 0으로 두면 자르지 않는다(예전 동작).
        site_cap = int(os.getenv("FL_SENSOR_MAX_WINDOWS_PER_SITE", "600"))

        # 로봇별 조회 상한은 사업장 상한에서 역산한다. 넉넉히 받아 두고
        # 합친 뒤에 고르게 줄이는 편이, 로봇 하나가 쉬고 있을 때도
        # 사업장 총량을 채울 수 있어 낫다.
        n_robots = max(1, len(self.telemetry_paths))
        default_per_robot = (max(50, site_cap // n_robots * 2)
                             if site_cap > 0 else 300)
        per_robot = int(os.getenv("FL_SENSOR_CIN_LIMIT_PER_ROBOT",
                                  str(default_per_robot)))
        wait_total = float(os.getenv("FL_SENSOR_ROUND_WAIT_SEC", "90"))
        poll       = float(os.getenv("FL_SENSOR_ROUND_POLL_SEC", "2.0"))

        # 라운드마다 "최근 N개"를 다시 긁으면 간격이 짧을 때 같은 데이터를
        # 반복 학습하게 된다. 읽은 지점(ct)을 기억해 그 뒤만 받는다.
        # FL_SENSOR_INCREMENTAL=0 으로 끄면 예전처럼 최근 N개를 받는다.
        incremental = os.getenv("FL_SENSOR_INCREMENTAL", "1") != "0"

        deadline = time.time() + wait_total
        next_log = 0.0
        cursors = dict(self.telemetry_cursor)

        while True:
            if incremental:
                got, cursors = amr_ingest.fetch_site_windows_since(
                    self.telemetry_paths, self.telemetry_cursor, per_robot)
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
                    return SKIP_ROUND
                if n == 0:
                    print(f"    ✗ {site} 로봇 텔레메트리가 비어 있다.")
                    print(f"      확인: {len(self.telemetry_paths)}개 컨테이너 "
                          f"({', '.join(self.robot_names)})")
                    print("      Isaac Sim이 이 MN-CSE에 등록·전송 중인지 보라.")
                    return None
                print(f"    ⚠ 윈도우 {n}개로 진행한다 (최소 {want_min}개 기대)")
                break

            if now >= next_log:
                print(f"    ⏳ 윈도우 {n}/{want_min} 수집 중 "
                      f"(남은 {deadline - now:.0f}s)")
                next_log = now + 5.0

            time.sleep(poll)

        # 쓰기로 한 데이터까지만 커서를 민다. 실패한 라운드의 데이터를
        # 건너뛰지 않기 위해, 여기까지 온 뒤에 갱신한다.
        if incremental:
            self.telemetry_cursor = cursors

        # 사업장 단위로 고르게 줄인다.
        if site_cap > 0 and len(got["signals"]) > site_cap:
            before = len(got["signals"])
            got = amr_ingest.balance_across_robots(got, site_cap)
            print(f"    ⓘ 사업장 상한 적용: {before} -> "
                  f"{len(got['signals'])}개 (로봇별 균등)")

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
                # 정상 주행이 없는 라운드. 이상 데이터로 학습하느니 빠진다.
                # IN-AE에는 수집 타임아웃이 있어 나머지 노드로 집계가 진행된다.
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