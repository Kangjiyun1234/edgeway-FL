# edgeway-FL

**Standards-based federated learning over oneM2M and ETSI MEC for multi-robot floor inspection**

Submission for the 2026 oneM2M International Developer Contest.

Autonomous mobile robots (AMRs) drive fixed loops across three warehouse sites.
Each site trains locally on its own normal driving and shares only model weights,
never raw telemetry. The orchestration runs entirely on standard oneM2M resources
(`<AE>`, `<container>`, `<contentInstance>`, `<subscription>`, `<accessControlPolicy>`)
following the direction of oneM2M TR-0084, and the network side uses ETSI MEC
services exposed by the MEC Sandbox.

The question the system answers is not only "is something wrong" but **"is it the
floor or the robot?"** — a distinction that matters because the two faults need
opposite responses.

---

## 1. Demo scenarios

| # | Scenario | What it shows |
| --- | --- | --- |
| 1 | Normal driving | Robots drive their loops; telemetry lands in each site's MN-CSE |
| 2 | Network anomaly → handover | MEC028 detects a degraded Wi-Fi AP; the robot is moved to another AP and data keeps flowing |
| 3 | Network anomaly → buffering | No AP is available; windows are queued on the MEP side and replayed on recovery, with zero loss |
| 4 | Federated learning in real time | Sites A and B train and FedAvg; site C never contributes and still receives the global model |
| 5 | Anomaly detection: floor vs robot | A second robot is sent into the same zone to tell a floor fault apart from a faulty robot |

Scenarios 2 and 3 share one goal: **the link may degrade, but the data must not be
lost and the FL round must not stall.**

---

## 2. System architecture

```
                    ┌──────────────────────────────────────┐
                    │  IN-CSE  (cloud)                     │
                    │    IN-AE — FL coordinator            │
                    │      cnt-fl-control   round commands │
                    │      cnt-global-model global weights │
                    │      cnt-local-updates  per-node     │
                    │      cnt-fl-inference   verdicts     │
                    └──────────────┬───────────────────────┘
                                   │ Mcc  (notification-driven)
         ┌─────────────────────────┼─────────────────────────┐
         │                         │                         │
  ┌──────┴───────┐          ┌──────┴───────┐          ┌──────┴───────┐
  │ mep-a        │          │ mep-b        │          │ mep-c        │
  │  MN-CSE      │          │  MN-CSE      │          │  MN-CSE      │
  │  MN-AE-1     │          │  MN-AE-2     │          │  MN-AE-3     │
  │  mec011/013/ │          │  ...         │          │  ...         │
  │  021/028/    │          │              │          │              │
  │  033/040/046 │          │              │          │              │
  └──────┬───────┘          └──────┬───────┘          └──────┬───────┘
         │ Mca                     │                         │
  siteA_AMR_1..4            siteB_AMR_1..2            siteC_AMR_1..4
  (Isaac Sim registers these AEs and publishes telemetry)
```

Three independent sites, one Multi-access Edge Platform (MEP) each, one central
IN-CSE in the cloud. Each MEP runs its own ACME MN-CSE plus the MEC services.

### Who owns what

| Owner | Resources | Created by |
| --- | --- | --- |
| **Isaac Sim** | Robot AEs `siteX_AMR_n` and their `telemetry` / `status` containers | The simulator, at startup |
| **This repository** | `IN-AE`, `MN-AE-1/2/3`, their containers, subscriptions and ACPs | `fl/setup_resources_standard.py` |
| **MEC Sandbox** | MEPs, CSEs, Wi-Fi access points, robot UEs, mobility | The `edgeway` scenario |

`setup_resources_standard.py` deliberately does **not** create the robot AEs. It
discovers them (`fu=1&ty=2`) and reports whether the expected count is registered,
so a missing simulator is diagnosed instead of silently papered over.

### Division of responsibility

The FL side **detects and decides**. It does not command robots.

```
MEC028 (WAI)  ──▶  MN-AE judges  ──▶  writes verdict to oneM2M
                                          │
                                          ▼
                              MEC Sandbox controller acts
                              (MOBILITY / NETWORK-CHARACTERISTICS-UPDATE)
```

Verdicts go to `cnt-fl-inference` (anomaly decisions) and `cnt-net-status`
(link state and buffering). The sandbox and Isaac Sim read those and move robots.

---

## 3. oneM2M resource mapping

```
IN-CSE   acme-cse-in-<sandbox>
└── IN-AE
    ├── cnt-fl-control            round commands        acp-fl-control
    ├── cnt-global-model          global weights        acp-global-model
    ├── cnt-local-updates                              acp-local-updates
    │   ├── cnt-mn1               site A drop-box       acp-dropbox-mn1
    │   ├── cnt-mn2               site B drop-box       acp-dropbox-mn2
    │   └── cnt-mn3               site C drop-box       acp-dropbox-mn3
    └── cnt-fl-inference          anomaly verdicts

MN-CSE   acme-cse-mn-mep-a-<sandbox>
├── siteA_AMR_1 … siteA_AMR_4     (Isaac Sim)
│   ├── telemetry                 38-channel windows
│   └── status
└── MN-AE-1
    ├── cnt-local-model           round cache           acp-cache-mn1
    └── cnt-net-status            MEC028 snapshots      acp-cache-mn1
```

| FL function | oneM2M resource |
| --- | --- |
| Round control | `<contentInstance>` under `IN-AE/cnt-fl-control` |
| Command delivery | `<subscription>` per MN-AE on `cnt-fl-control` (NOTIFY) |
| Global weight distribution | `<contentInstance>` under `cnt-global-model` |
| Local update upload | `<contentInstance>` under `cnt-local-updates/cnt-mnX` |
| Update collection | `<subscription>` per drop-box, target IN-AE |
| Sensor source | `<contentInstance>` under each robot's `telemetry` |
| Access control | `<accessControlPolicy>` + `acpi`, per-node scoped |

Each drop-box carries its own ACP so one site cannot read another site's updates.
`pvs` is scoped as tightly as `pv`; otherwise any originator could rewrite the policy.

### Standards conformance notes

- `X-M2M-RI` is set on **every** request with a process-unique value, as TS-0009
  requires for the HTTP binding.
- `X-M2M-RVI` is sent (release 3).
- `cnf` is set to `application/json:0` whenever a dict is serialized into `con`,
  so a resource states its own content type.
- Dual-CSE routing uses an `@in/` path prefix internally; `onem2m_utils._url()`
  resolves it to the central CSE, everything else to the bound MN-CSE.

---

## 4. Federated learning design

### Participation

| Site | Robots | Role |
| --- | ---: | --- |
| A (`mn1`, mep-a) | 4 | Trains, contributes to FedAvg |
| B (`mn2`, mep-b) | 2 | Trains, contributes to FedAvg |
| C (`mn3`, mep-c) | 4 | **Never contributes.** Receives the global model, fine-tunes locally |

Site C is the point of the demo. It joins late, has little data of its own, and
must still detect faults it has never seen — using a model trained on two other
sites' floors. FedAvg is weighted by sample count.

Robots within a site are pooled: however many robots a site runs, their windows
become one site dataset, because a fault in the floor belongs to the site, not
to a robot.

### Training signal

**Only normal driving is used for training.** There are no fault labels anywhere
in the training path. Anomaly labels (`nav == "over_bump"`) are used solely to
score detection afterwards.

Stopped windows are excluded by **speed**, not by the navigation label — using the
label would leak the answer into the input.

### Model: conditional 1D CNN VAE

`fl/model_cvae.py` — `CVAE1D`, 90,806 parameters.

```
input    (B, 38, 60)      38 channels × 60 samples @ 60 Hz = 1 s window
cond     (B, 8)           5 navigation groups + 3 speed bins
encoder  60 → 30 → 15 → 8
latent   24
decoder  8 → 15 → 30 → 60
loss     ELBO = reconstruction MSE + β·KL   (β = 0.5)
```

The condition vector matters: the same vibration means something different when
turning at speed than when creeping into a load bay. Without conditioning, normal
cornering reads as a fault.

Evaluation uses `z = mu` rather than sampling, so thresholds are reproducible.

Channels (38): 3 IMUs × 8 (`ax ay az gx gy gz roll pitch`) + 14 encoder channels
(drive `pos/vel/eff`, swivel `pos`, 4 rollers × `pos/vel`, lift `pos/eff`).
Position channels are window-mean-subtracted — absolute position is not a fault signal.

### Anomaly scoring

Per-channel reconstruction error is normalized by that channel's mean error on the
site's own calibration windows, then averaged. Per-site normalization is necessary:
each site's floor and each robot's build have a different "normal", and a shared
statistic makes normal look anomalous.

Threshold: `mean + 3σ` over normal calibration windows.

### Scenario 5: floor fault or robot fault

```
robot 1 reports an anomaly in zone Z
        │
        ▼
send robot 2 into zone Z
        │
   ┌────┴────┐
   ▼         ▼
also      no anomaly
anomalous      │
   │           ▼
   ▼      robot 1 is faulty
floor     → move it to the standby area
fault
```

The verdict is written to `cnt-fl-inference`. Dispatching the second robot and
moving the faulty one are sandbox actions, not FL actions.

---

## 5. Network scenarios over ETSI MEC

Each MEP runs `mec028-1` (`meep-wais`), AdvantEDGE's implementation of
**ETSI GS MEC 028** (WLAN Access Information API).

### Observation — MEC028

```
GET /<sandbox>/<mep>/wai/v2/queries/sta/sta_information
GET /<sandbox>/<mep>/wai/v2/queries/ap/ap_information
```

| Field | Use |
| --- | --- |
| `staInfo.apAssociated.bssid` | which AP the robot is on |
| `staInfo.rssi.rssi` | signal strength (dBm) |
| `staInfo.staDataRate.staLastDataUplinkRate` | uplink rate (kbps) |
| `staInfo.neighborReport[].bssid` | APs the robot can actually hear |
| `apInfo.bssLoad.channelUtilization` | channel occupancy (255 = 100 %) |
| `apInfo.bssLoad.staCount` | associated stations |

`fl/mec_wai.py` turns these into `GOOD` / `DEGRADED` / `LOST` with the reasons
attached, so the dashboard can show *why* a link was judged bad.

### Action — MEC Sandbox controller

```
POST /<sandbox>/sandbox-ctrl/v1/events/MOBILITY
POST /<sandbox>/sandbox-ctrl/v1/events/NETWORK-CHARACTERISTICS-UPDATE
```

`fl/mec_sandbox.py` wraps these: degrade an AP, disconnect a robot, hand it over,
restore. MEC028 names APs by BSSID while the controller names them by element
name, so `fl/mec_topology.py` holds the translation table (regenerable from the
scenario YAML via `export_from_yaml()`).

### Scenario 2 — handover

Degrade the serving AP → detect through MEC028 → pick a target (the robot's own
neighbor report first, since that is evidence the robot can hear it; otherwise the
least-loaded AP in the same site; `spare` site last, because leaving the site
changes MEP and would need MEC021 AMS) → request MOBILITY → confirm the new AP.

Staying inside the site keeps the same MN-CSE, so only the path changes.

### Scenario 3 — store-and-forward buffering

While the link is down, windows are written to a disk queue
(`fl/net_scenarios.py:StoreAndForward`) and replayed in order on recovery.

Disk rather than memory, and `os.replace` for every write, so a crash mid-outage
still leaves a complete queue — otherwise "zero loss" would not hold. A failed
send stops the drain instead of skipping ahead, preserving order.

### Topology (`edgeway` scenario)

| Site | MEP | Wi-Fi APs | Robots |
| --- | --- | ---: | ---: |
| site-a | mep-a | 7 | 4 |
| site-b | mep-b | 7 | 2 |
| site-c | mep-c | 10 | 4 |
| spare | mep-spare | 13 | 0 (handover refuge) |

---

## 6. Model weight transport over oneM2M

Weights travel **inside oneM2M**, not beside it. `fl/model_codec.py` encodes a
`state_dict` as fp16 → gzip → base64 and puts it in `contentInstance.con`, with a
SHA-256 of the transmitted bytes for integrity.

```
CVAE1D (90,806 parameters)
    fp32 raw      ≈ 474 KB base64
    fp16 + gzip   ≈ 217 KB base64     ← default
```

One contentInstance is enough on ACME. Chunking (`FL_CHUNK_WEIGHTS`) exists for
CSEs with a small payload ceiling: fragments go into a dedicated child container
and the parent CIN carries only a manifest, so a subscription on the parent fires
once rather than once per fragment. Verified at 39 fragments (0.8 s upload,
0.1 s retrieval).

---

## 7. Project structure

```
fl/
  config.py                  per-MEP endpoints, model/training config, node roles
  amr_schema.py              38-channel spec, condition vector, window parsing
  model_cvae.py              CVAE1D
  model_codec.py             weight codec (fp16+gzip+base64, sha256, chunking)
  edge_node.py               local training, calibration, scoring
  aggregator.py              FedAvg
  onem2m_utils.py            oneM2M client, dual-CSE routing, discovery
  amr_ingest.py              CIN → training windows (per robot, per site)
  prepare_data_amr.py        offline JSONL → pkl
  setup_resources_standard.py   resource/ACP/subscription setup, AE verification
  in_ae_standard.py          IN-AE coordinator
  mn_ae_standard.py          MN-AE site trainer
  mec_topology.py            AP↔BSSID, robot AE↔UE↔MAC tables
  mec_wai.py                 MEC028 client and link judgement
  mec_sandbox.py             MEC Sandbox controller events
  net_scenarios.py           scenarios 2 and 3
  personalize.py             local fine-tuning for the non-contributing node
  dashboard_server.py        dashboard backend

fl_bearing_dashboard.html    dashboard front end
FL-diagrams/                 PlantUML sequence/structure diagrams
out/FL-diagrams/             rendered diagrams
archive/femto/               previous FEMTO bearing demo, kept for reference
archive/legacy/              superseded entry points
env.local.example            environment template (copy to .env.local)
run_fl.sh                    start the whole demo
clean_fl.sh                  reset resources and local artifacts
```

---

## 8. Configuration

Everything is environment-overridable; defaults live in `fl/config.py`.

### Sandbox

The sandbox address is **not** stored in this repository. Copy the template and
fill it in:

```bash
cp env.local.example .env.local   # .gitignore keeps .env* out of git
# fill in FL_SANDBOX_HOST and FL_SANDBOX_NAME
source .env.local
```

Anything that needs the sandbox calls `config.require_sandbox()` first and stops
with instructions if it is unset. Offline work (`prepare_data_amr.py`) does not
need it and runs regardless.

| Variable | Default | Meaning |
| --- | --- | --- |
| `FL_SANDBOX_HOST` | **required** | host the sandbox is deployed on |
| `FL_SANDBOX_NAME` | **required** | sandbox instance name |
| `FL_SANDBOX_SCHEME` | `https` | scheme |
| `FL_TLS_VERIFY` | `1` | set `0` for a self-signed sandbox certificate |
| `FL_IN_BASE_URL` / `FL_IN_CSE_NAME` | derived | override the central CSE |
| `FL_MN_BASE_URL_MN1` … | derived | override one site's CSE |

Derived endpoints, with `<host>` and `<sandbox>` taken from the environment:

```
IN-CSE   https://<host>/<sandbox>/meep-acme-in-cse          acme-cse-in-<sandbox>
mep-a    https://<host>/<sandbox>/mep-a/meep-acme-mn-cse    acme-cse-mn-mep-a-<sandbox>
MEC028   https://<host>/<sandbox>/mep-a/wai/v2
control  https://<host>/<sandbox>/sandbox-ctrl/v1
```

### Federated learning

| Variable | Default | Meaning |
| --- | --- | --- |
| `FL_LOCAL_ONLY_NODES` | `mn3` | receives the global model, never contributes |
| `FL_COLD_START_NODES` | — | excluded from training entirely |
| `FL_SENSOR_FROM_CSE` | `1` | `0` falls back to local pkl (offline) |
| `FL_SENSOR_MIN_WINDOWS` | `40` | minimum windows before a round proceeds |
| `FL_SENSOR_CIN_LIMIT_PER_ROBOT` | `150` | CINs fetched per robot per round |
| `FL_MAX_CIN_BYTES` | 4 MB | embed ceiling before falling back |
| `FL_CHUNK_WEIGHTS` | `0` | fragment weights across CINs |
| `FL_CREATE_AES` | `0` | robot AEs come from Isaac Sim |

Model and training defaults: `seq_len=60`, `n_channels=38`, `cond_dim=8`,
`latent_dim=24`, `hidden=48`, `beta=0.5`; 10 global rounds, 10 local epochs,
batch 32, lr 1e-3.

### MEC028 judgement thresholds

| Variable | Default | Meaning |
| --- | --- | --- |
| `FL_RSSI_WARN_DBM` | `-75` | below this, warn |
| `FL_RSSI_BAD_DBM` | `-85` | below this, bad |
| `FL_UL_MIN_KBPS` | `500` | minimum uplink rate |
| `FL_CHU_WARN_PCT` | `70` | channel occupancy ceiling |

**These are estimates, not measurements.** See Known limitations.

---

## 9. How to run

### Prerequisites

1. The MEC Sandbox `edgeway` scenario is deployed (3 MEPs + cloud IN-CSE).
2. Isaac Sim has registered the robot AEs and is publishing telemetry.
3. `FL_SANDBOX_HOST` and `FL_SANDBOX_NAME` are set (see Configuration).

```bash
pip install -r requirements.txt
cp env.local.example .env.local   # then fill it in
source .env.local
```

### Check the environment first

```bash
python3 fl/setup_resources_standard.py --verify   # all four CSEs, resources, robot AEs
python3 fl/mec_wai.py --nodes mn3                 # MEC028 link state
```

`--verify` reports missing robot AEs as a warning: those belong to Isaac Sim, so
the fix is to start the simulator, not to create them here.

### Set up resources

```bash
python3 fl/setup_resources_standard.py            # create
python3 fl/setup_resources_standard.py --clean    # delete ours, recreate
python3 fl/setup_resources_standard.py --nodes mn1,mn2
```

`--clean` removes only resources this repository owns. Robot AEs are never touched.

### Run the federated learning demo

```bash
./run_fl.sh                           # resources → 3 MN-AEs → IN-AE → dashboard
FL_SENSOR_FROM_CSE=0 ./run_fl.sh      # offline: local pkl instead of the CSE
```

Or individually:

```bash
python3 fl/mn_ae_standard.py 0 5001   # MN-AE-1  site A  mep-a
python3 fl/mn_ae_standard.py 1 5002   # MN-AE-2  site B  mep-b
python3 fl/mn_ae_standard.py 2 5003   # MN-AE-3  site C  mep-c
python3 fl/in_ae_standard.py          # coordinator
python3 fl/dashboard_server.py        # http://localhost:7000
```

Start the MN-AEs **before** the IN-AE. The MN-AEs must have their `cnt-fl-control`
subscriptions in place; otherwise the first round command notifies nobody.

### Run the network scenarios

```bash
python3 fl/net_scenarios.py watch   --node mn3 --seconds 60
python3 fl/net_scenarios.py s2      --node mn3 --robot siteC_AMR_2
python3 fl/net_scenarios.py s3      --node mn3 --robot siteC_AMR_2 --outage 40
python3 fl/net_scenarios.py restore --node mn3
```

`--no-inject` skips fault injection and only detects and reacts — use it when the
fault is injected from the sandbox UI instead. **Always finish with `restore`**;
otherwise degraded APs persist into the next run.

### Offline data preparation

```bash
AMR_DATA=/path/to/telemetry python3 fl/prepare_data_amr.py
```

Reads `<robot>/telemetry.json` (JSON Lines, one line per contentInstance) and
assigns robots to sites by the `siteA_*` / `siteB_*` / `siteC_*` prefix, or by
`AMR_ROBOT_MAP`.

---

## 10. Ports

| Component | Port |
| --- | ---: |
| MN-AE-1 notification server | 5001 |
| MN-AE-2 notification server | 5002 |
| MN-AE-3 notification server | 5003 |
| IN-AE notification server | 6000 |
| Dashboard | 7000 |

`node_id` and port must stay aligned: `0 → mn1 → 5001`, `1 → mn2 → 5002`,
`2 → mn3 → 5003`.

---

## 11. Known limitations

Stated plainly, because the measurements are what they are.

**Federated learning has not yet been shown to beat the simpler baseline.**
On the current site C data, FedAvg(A,B) scores AUROC 0.5925 for floor-bump
detection while a model-free z-score baseline scores 0.6150. Local fine-tuning on
site C makes it worse (0.59 → 0.51) because site C has far too little data to
fine-tune on.

**Site C is data-starved.** Only one robot has usable data, 25 moving windows.
The design calls for four. Normal driving at site A's scale (~400 windows) and
50+ bump samples are needed before the comparison means anything.

**Floor faults are more detectable than robot faults.** A single channel
(`roller_rl.vel`) reaches AUROC 0.855 on floor bumps versus about 0.71 for robot
faults — so scenario 5's floor side is on firmer ground than its robot side.

**The MEC028 thresholds in section 8 are estimates.** They have not been checked
against a running sandbox. If normal RSSI sits outside the assumed range,
scenario 2 will either miss the degradation or flag healthy links.

**The dashboard still renders the previous bearing-demo charts.** It has not been
reworked for AMR telemetry.

**The sandbox scenario YAML lists four robots at site B** while the code and data
use two (`ROBOTS_PER_NODE`). The YAML is being corrected.

---

## 12. Standards referenced

| Standard | Use |
| --- | --- |
| oneM2M TS-0001 | resource types, ACP semantics |
| oneM2M TS-0004 | AE registration attributes (`rr`, `srv`, `api`) |
| oneM2M TS-0009 | HTTP binding, mandatory `X-M2M-RI` |
| oneM2M TR-0084 | federated learning over oneM2M resources |
| ETSI GS MEC 011 | service registry (app enablement) |
| ETSI GS MEC 013 | location service |
| ETSI GS MEC 021 | application mobility (cross-MEP handover) |
| ETSI GS MEC 028 | WLAN access information — scenarios 2 and 3 |
| ETSI GS MEC 033 | IoT service |

CSE: [ACME](https://github.com/ankraft/ACME-oneM2M-CSE).
Network emulation: [AdvantEDGE / ETSI MEC Sandbox](https://github.com/InterDigitalInc/AdvantEDGE).
