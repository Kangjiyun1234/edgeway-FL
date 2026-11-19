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

| # | Scenario | What it shows | ETSI MEC API |
| --- | --- | --- | --- |
| 1 | Normal driving | Robots drive their loops; telemetry lands in each site's MN-CSE | MEC 033 (site app finds its MN-CSE) + MEC 011 (registers as a MEC app, discovers 013/028) |
| 2 | Network anomaly → handover | MEC028 detects a degraded Wi-Fi AP; the robot is moved to another AP and data keeps flowing | MEC 028 (observe) + sandbox controller (act) |
| 3 | Network anomaly → buffering | No AP is available; windows are queued on the MEP side and replayed on recovery, with zero loss | MEC 028 (observe) + sandbox controller (act) |
| 4 | Federated learning in real time | Sites A and B train and FedAvg; site C never contributes and still receives the global model | MEC 028 + MEC 013, read once per round into `cnt-net-status` |
| 5 | Anomaly detection: floor vs robot | A second robot is sent into the same zone to tell a floor fault apart from a faulty robot | MEC 013 + MEC 028, attached to each verdict as `mec` |

The sandbox controller (`sandbox-ctrl`) is the MEC Sandbox's own control API, not
an ETSI MEC specification, so it carries no GS number. Section 12 lists every MEC
service on the MEPs and which of them this repository actually calls.

Scenarios 2 and 3 share one goal: **the link may degrade, but the data must not be
lost and the FL round must not stall.**

---

## 2. System architecture

Three warehouse sites, one Multi-access Edge Platform (MEP) each. Every MEP runs
its own ACME MN-CSE next to the MEC services. The IN-CSE is **not** part of the
sandbox — it is a shared CSE operated outside it, reached over Mca.

```
                IN-CSE   shared, outside the sandbox
                ┌──────────────────────────────────────────────┐
                │  IN-AE               FL coordinator          │
                │  TRAINING-AE-1/2/3   site trainers, 2nd reg. │
                └──────────────────────┬───────────────────────┘
                                       │ Mca   HTTPS + <pollingChannel>
  ┌────────────────────────────────────┴────────────────────────────────────┐
  │  MEC Sandbox — scenario `edgeway`                                        │
  │                                                                          │
  │   site-a / mep-a         site-b / mep-b         site-c / mep-c           │
  │   ┌────────────────┐     ┌────────────────┐     ┌────────────────┐       │
  │   │ MN-CSE         │     │ MN-CSE         │     │ MN-CSE         │       │
  │   │  TRAINING-AE-1 │     │  TRAINING-AE-2 │     │  TRAINING-AE-3 │       │
  │   │  ANALYSIS-AE-1 │     │  ANALYSIS-AE-2 │     │  ANALYSIS-AE-3 │       │
  │   │  siteA_AMR_1-4 │     │  siteB_AMR_1-2 │     │  siteC_AMR_1-2 │       │
  │   │ MEC 011 013 021│     │       ...      │     │       ...      │       │
  │   │ 028 033 040 046│     │                │     │                │       │
  │   └────────────────┘     └────────────────┘     └────────────────┘       │
  │     poa-wifi 1..7          poa-wifi 1..7          poa-wifi 1..10         │
  │     robot-a1..a4 UE        robot-b1..b4           robot-c1..c4           │
  │                                      spare / mep-spare — 13 APs, no robots│
  └──────────────▲───────────────────────────────────────▲──────────────────┘
                 │ Mca                                   │ Mca
          Isaac Sim host                          FL processes, one host
          registers robot AEs,                    3 site trainers + coordinator
          publishes telemetry                     + dashboard
```

The three site MEPs each run the same seven ETSI MEC services (`mec011-N` …
`mec046-N` in the scenario); `mep-spare` runs none. This repository calls
**MEC 011, 013, 028 and 033**; the others are deployed but not used here — see
section 12.

### Resources are not processes

A `<AE>` resource in a CSE is a registration, not a running program. None of this
repository's code runs *inside* a CSE or a MEP: the three site trainers, the
coordinator and the dashboard are ordinary Python processes on one host, and they
talk to the CSEs over Mca. Isaac Sim runs on a second host and reaches the MN-CSEs
the same way.

### Why the site trainer is registered on two CSEs

`TRAINING-AE-i` appears under its own MN-CSE **and** under the IN-CSE. That is not
duplication. They are two registrations of one application, and both are needed.

On the **MN-CSE** it is the site's training app. It subscribes to the robots'
`telemetry` containers, caches the round's model under `cnt-local-model`, and
records link state in `cnt-net-status`. All of that stays inside the site.

On the **IN-CSE** it is the identity the coordinator grants rights to. Two things
depend on that registration:

1. **Access control needs a name to grant rights to.** `acp-dropbox-mnX` gives
   CREATE to exactly one originator, `CTRAINING-AE-X`. An originator that is not a
   registered AE of that CSE cannot hold rights on it, so without the registration
   a site could not upload its weights at all — and the per-site isolation that
   stops one site from reading another's updates would have nothing to name.

2. **The AE is request-unreachable.** The FL processes sit behind a campus
   firewall, so the IN-CSE cannot call them. TS-0001 covers exactly this case with
   `<pollingChannel>`: the AE holds a long poll open and the CSE delivers
   notifications on that response. A `<pollingChannel>` is a child of an `<AE>`,
   and **only that AE may create it** — ACME refuses even the admin originator
   (`AE.py`: "Check correct originator. Even the ADMIN is not allowed that").
   So the AE has to exist on the IN-CSE before a single round command can be
   delivered.

`setup_resources_standard.py` creates both registrations. When it does not, the
failure is silent in the worst way: subscriptions are accepted, no notification
ever arrives, and every site waits out its full round budget.

#### Why not relay over Mcc instead

Each MN-CSE *is* registered to the IN-CSE — every one of them carries a
`<remoteCSE>` for it (`cst=1`, `rr=true`). A reader may reasonably ask why the FL
traffic does not simply ride that link, with the sites addressing the IN-CSE
through their own registrar.

It does not, for a reason that survives whether or not retargeting works: the
`<pollingChannel>` has to live on the CSE that delivers the notification. The
round command is a `<contentInstance>` created under `IN-AE/cnt-fl-control` on the
IN-CSE, so the IN-CSE is the CSE that notifies — and the only place it can leave a
notification for an unreachable AE is a polling channel under that AE, on itself.
Relaying the *request* through a registrar would not move the *notification*.

So the sites hold an Mca connection to each CSE directly, and the Mcc link is left
to do what CSE-to-CSE registration does on its own. We also could not confirm the
matching `<remoteCSE>` on the IN-CSE side from our originator — discovery returned
none — but nothing in this design depends on it either way.

### Who owns what

| Owner | Resources | Created by |
| --- | --- | --- |
| **Isaac Sim** | Robot AEs `siteX_AMR_n` and their `telemetry` / `status` containers | The simulator, at startup |
| **This repository** | `IN-AE`, `TRAINING-AE-1/2/3`, `ANALYSIS-AE-1/2/3`, their containers, subscriptions and ACPs | `fl/setup_resources_standard.py` |
| **MEC Sandbox** | MEPs, CSEs, Wi-Fi access points, robot UEs, mobility | The `edgeway` scenario |

`setup_resources_standard.py` deliberately does **not** create the robot AEs. It
finds their `telemetry` containers by label discovery
(`fu=1&ty=3&lbl=siteX:telemetry`, falling back to `lbl=site:siteX` + `type:telemetry`),
re-reads each hit to confirm its labels, and reports whether the expected count is
registered, so a missing simulator is diagnosed instead of silently papered over.
The site trainers use the same discovery at startup. When nothing is found under
the labels they fall back to the `siteX_AMR_n/telemetry` naming rule.

The combined label matters because ACME treats several `lbl` values in one query
as **OR**: asking for `site:siteA` and `type:telemetry` together also returns this
repository's own containers that carry either label.

### Division of responsibility

The FL side **detects and decides**. It does not command robots.

```
MEC028 (WAI)  ──▶  the site judges  ──▶  writes a verdict to oneM2M
                                              │
                                              ▼
                                  MEC Sandbox controller acts
                                  (MOBILITY / NETWORK-CHARACTERISTICS-UPDATE)
```

Floor verdicts go to `ANALYSIS-AE-i/cnt-verdict` on the site's own MN-CSE and
never leave the site; link state and buffering go to `cnt-net-status`. The
sandbox and Isaac Sim read those and move robots.

Writing the verdict is the analysis app's job, not the trainer's. They are
separate AEs with separate ACPs because the design separates the two roles: the
robots may *read* verdicts and be notified of them, but only the analysis app may
create one.

---

## 3. oneM2M resource mapping

Both trees below were read back from the running CSEs, not written from the
design. `<in-cse>` and `<sandbox>` stand for values that live in `.env.local`.

```
IN-CSE   <in-cse>                            outside the sandbox, shared
│
├── CAdmin                   <AE>   built into the CSE — never delete
├── acpCreateRootResources   <ACP>  built in
├── acpRetrieveCSEBase       <ACP>  built in
│
├── lightbulbAE …            <AE>   other teams' resources on the shared CSE — never touched
│
├── acp-fl-control           <ACP>  IN-AE all · TRAINING-AE-* RETRIEVE+NOTIFY
├── acp-global-model         <ACP>  same
├── acp-local-updates        <ACP>  IN-AE all · TRAINING-AE-* CREATE
├── acp-dropbox-mn1/2/3      <ACP>  IN-AE all · that one site only, CREATE
│
├── TRAINING-AE-1            <AE>   CTRAINING-AE-1   site A trainer
│   └── pch                  <pollingChannel>   receives round commands
├── TRAINING-AE-2            <AE>   CTRAINING-AE-2
│   └── pch                  <pollingChannel>
├── TRAINING-AE-3            <AE>   CTRAINING-AE-3
│   └── pch                  <pollingChannel>
│
└── IN-AE                    <AE>   CIN-AE           coordinator
    ├── pch-in-ae            <pollingChannel>   receives drop-box notifications
    │
    ├── cnt-fl-control       <CNT>  mni=50    round commands   IN → sites
    │   ├── sub_mn1/2/3      <SUB>  net=3, one per site trainer
    │   └── <CIN>  {jobState, currentRound, maxRounds, timestamp}
    │
    ├── cnt-global-model     <CNT>  mni=50    global weights   IN → sites
    │   └── <CIN>  fp16+gzip+base64 state_dict
    │
    ├── cnt-local-updates    <CNT>  mni=50
    │   ├── cnt-mn1          <CNT>  mni=50    site A drop-box
    │   │   ├── sub_in_dropbox_mn1  <SUB>  net=3, target IN-AE
    │   │   └── <CIN>  {type, round, train_loss, val_loss, val_auroc,
    │   │               num_samples, weights, window_scores,
    │   │               window_labels, threshold}
    │   ├── cnt-mn2          <CNT>  site B drop-box
    │   │   └── sub_in_dropbox_mn2  <SUB>  net=3, target IN-AE
    │   └── cnt-mn3          <CNT>  site C — uploads fl-inference only, no weights
    │       └── sub_in_dropbox_mn3  <SUB>  net=3, target IN-AE
    │
    └── cnt-fl-inference     <CNT>  mni=200   per-round model evaluation
        └── <CIN>  {type:"global-eval", round, model, timestamp, auroc,
                    baseline_auroc, delta_vs_baseline, detected, false_alarms,
                    threshold, threshold_mean3s, mean_normal, mean_anomaly,
                    separation, n_normal, n_anomaly, by_fault}
```

```
MN-CSE   acme-cse-mn-mep-a-<sandbox>          inside the sandbox, one per MEP
│
├── CAdmin / acpCreateRootResources / acpRetrieveCSEBase    built in
├── <in-cse-id>              <remoteCSE>   registrar link, cst=1 (IN-CSE)
│
├── siteA_AMR_1 … _4         <AE>   ADN-AE, registered by Isaac Sim
│   ├── telemetry            <CNT>  mni=10000
│   │      lbl = site:siteA · robot:AMR_1 · ae:siteA_AMR_1
│   │            · type:telemetry · siteA:telemetry
│   │   ├── sub-sensor-data-mn1   <SUB>  net=3 → the trainer's poa
│   │   └── <CIN>  {robot, seq, t_wall, t_sim, nav,
│   │               pose{x,y,yaw}, 38 signal channels, label}
│   └── status               <CNT>
│
├── acp-cache-mn1            <ACP>  CTRAINING-AE-1 all · CIN-AE RETRIEVE+DISCOVERY
├── acp-verdict-mn1          <ACP>  CANALYSIS-AE-1 all
│                                   · CTRAINING-AE-1 RETRIEVE+DISCOVERY
│                                   · robot AEs RETRIEVE+NOTIFY+DISCOVERY
│
├── TRAINING-AE-1            <AE>   CTRAINING-AE-1   training app
│   │                               poa=http://<host>:5001, rr=true
│   │                               local notifications are direct by default,
│   │                               so there is no <pollingChannel> here
│   ├── cnt-local-model      <CNT>  mni=5     round cache
│   └── cnt-net-status       <CNT>  mni=100   MEC028 snapshots, buffer state
│
└── ANALYSIS-AE-1            <AE>   CANALYSIS-AE-1   analysis app
    └── cnt-verdict          <CNT>  mni=500   stays inside the site
        └── <CIN>  {type:"amr-verdict", site, node, robot, verdict:"ANOMALY",
                    score, threshold, ratio, consecutive, window_index,
                    observed_at, pose{x,y}, round, model, timestamp}
```

mep-b and mep-c have the same structure apart from the robot count (4 / 2 / 2)
and the node suffix. Site C (`FL_LOCAL_ONLY_NODES=mn3`) still trains locally and
still writes verdicts to `ANALYSIS-AE-3/cnt-verdict`; it differs only in *what it
uploads*:

```
site A / B  →  cnt-mn1, cnt-mn2   {type:"fl-update", weights, train_loss, num_samples, ...}
                                   IN-AE puts them into FedAvg
site C      →  cnt-mn3            {type:"fl-inference", train_loss, num_samples, ...}
                                   no weights; IN-AE does not wait for it and only
                                   logs it as [NO-AGG/NOTIFY]
```

| FL function | oneM2M resource |
| --- | --- |
| Round control | `<contentInstance>` under `IN-AE/cnt-fl-control` |
| Command delivery | `<subscription>` per site on `cnt-fl-control` (NOTIFY) |
| Notification to a firewalled AE | `<pollingChannel>` under each AE on the IN-CSE |
| Global weight distribution | `<contentInstance>` under `cnt-global-model` |
| Local update upload | `<contentInstance>` under `cnt-local-updates/cnt-mnX` |
| Update collection | `<subscription>` per drop-box, target IN-AE |
| Sensor source | `<contentInstance>` under each robot's `telemetry` |
| Access control | `<accessControlPolicy>` + `acpi`, per-node scoped |
| Per-round model evaluation | `<contentInstance>` under `cnt-fl-inference` |
| Floor verdict | `<contentInstance>` under `ANALYSIS-AE-i/cnt-verdict` (MN-CSE) |

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
| C (`mn3`, mep-c) | 2 | **Never contributes.** Receives the global model, fine-tunes locally |

Site C is the point of the demo. It joins late, has little data of its own, and
must still detect faults it has never seen — using a model trained on two other
sites' floors.

#### Aggregation weighting: equal per site, not per sample

Textbook FedAvg weights each client by its sample count, because the goal is to
approximate training on the pooled data. This demo deliberately does not.

The participants here are **sites**, and sites run different numbers of robots
(A has four, B has two). Robots publish continuously, so in any given round a
four-robot site accumulates roughly twice the windows of a two-robot site. Under
sample-count weighting that difference becomes influence: the larger site pulls
the shared model twice as hard. Fleet size is a fact about a warehouse's budget,
not a claim to a louder voice.

So every site carries the same weight, regardless of how much it contributed
that round. This is the usual choice in cross-silo federated learning, where
each participant is an institution rather than a device. It equalizes influence
without discarding anyone's data — the alternative, trimming every site down to
the smallest one's volume, throws away real observations to achieve the same
balance.

Concretely, with site A holding a model parameter at 10 and site B at 0:

| Weighting | A's share | Result |
| --- | ---: | ---: |
| sample count (240 vs 120) | 67 % | 6.67 |
| **equal per site** | **50 %** | **5.00** |

A node that trained on nothing in a round (no normal windows arrived) is dropped
from the average rather than counted as a zero-weight participant.

Set `FL_AGG_WEIGHTING=samples` to use textbook FedAvg instead. Every round logs
which weighting was applied and the resulting per-node weights.

#### Keeping the data balanced

Two caps exist so that one site's volume does not quietly dominate, each at a
different stage:

| Setting | Stage | Default |
| --- | --- | --- |
| `FL_SENSOR_MAX_WINDOWS_PER_SITE` | live collection, per round | 50 |
| `AMR_MAX_NORMAL_PER_SITE` | offline preprocessing | off |

The live cap takes the **oldest** windows a site holds, in `t_wall` order, and
carries the remainder into the next round rather than discarding it
(`amr_ingest.sort_by_time` + `take_head`). Time order matters because a round is
meant to be a slice of driving, not a sample of it; and carrying the surplus
forward means a site that collects faster than it trains does not lose data, it
just runs a round behind.

An earlier version spread the cap evenly across a site's robots
(`balance_across_robots`). That function is still in the file but nothing calls
it: robots within a site are pooled anyway, so which robot a window came from
does not change what the site learns.

The offline cap is off by default because `min` silently discards most of the
data when one site's collection is incomplete. Enabling it when the smallest
site holds less than `AMR_MIN_BALANCE_RATIO` (0.5) of the largest is refused
with an explanation rather than applied.

Robots within a site are pooled: however many robots a site runs, their windows
become one site dataset, because a fault in the floor belongs to the site, not
to a robot.

### Training signal — unsupervised by default

Training sees **no labels at all**. A robot on a real floor does not announce
which of its windows are faulty, so neither does the training path here. Labels
exist only in the fixed validation set, and only for scoring.

That leaves the training set contaminated: whatever anomalies the robots drove
over are in it. Unsupervised anomaly detection expects this — most methods carry a
contamination parameter rather than assuming clean data — and the measurements
below show the detector tolerating it: trained on a 27 %-contaminated stream, the
ratio of anomalous to normal reconstruction error still widens from 1.27 to 2.46
over sixty epochs.

**Trimmed loss** is the textbook way to clean a contaminated training set without
labels: each epoch, the highest-loss fraction of a batch is left out of the
backward pass, on the assumption that anomalies are the high-loss samples. It is
the trimmed mean of statistics and least-trimmed-squares of regression.

```python
per   = model.loss_terms(xb, cb)          # per-sample ELBO
keep  = per.topk(int(bs * (1 - trim)), largest=False).values
loss  = keep.mean()
```

**It is off by default (`FL_TRIM_RATIO=0`), because it improves ranking and ruins
the verdict.** Trained on the second capture, scored against the shipped
validation set, five seeds:

| trim | AUROC (10-round mean) | false alarms at R10 (of 318 normal) |
| ---: | ---: | ---: |
| **0** | 0.7796 | **138** |
| 0.05 | 0.7818 | 175 |
| 0.10 | 0.7849 | 185 |
| 0.15 | 0.7919 | 212 |
| 0.25 | 0.7976 | 244 |

Both columns are monotone in the trim ratio and the spread is ±0.002, so the
trade-off is real rather than noise. AUROC only ranks; the verdict has to cross an
absolute threshold, and that is what breaks.

The mechanism is measurable. Trimming narrows what the model generalizes to: its
score on its *own* training set hardly moves (median 1.27 → 1.30 at 25 % trim)
while its score on unseen normal driving nearly triples (4.12 → 11.79). The
threshold is calibrated on the training set, so it rises only slightly
(5.47 → 5.98) and cannot follow. Anomaly scores rise too, which is why the ranking
improves even as the alarm becomes useless.

Only the backward pass is trimmed; calibration still sees every training window.

`FL_TRIM_WARMUP_EPOCHS` (2) skips trimming at the start, since an untrained model
has high loss everywhere and cutting then would discard the wrong windows.

Stopped windows are excluded by **speed**, not by the navigation label — using the
label would leak the answer into the input.

### Model: conditional 1D CNN VAE

`fl/model_cvae.py` — `CVAE1D`, 90,806 parameters.

```
input    (B, 38, 60)      38 channels × 60 samples @ 60 Hz = 1 s window
cond     (B, 11)          5 navigation groups + 3 speed bins + 3 load states
encoder  60 → 30 → 15 → 8
latent   24
decoder  8 → 15 → 30 → 60
loss     ELBO = reconstruction MSE + β·KL   (β = 0.5)
```

The condition vector matters: the same vibration means something different when
turning at speed than when creeping into a load bay, or when the robot is empty
rather than carrying a heavy load. Without conditioning, normal cornering reads as
a fault.

Evaluation uses `z = mu` rather than sampling, so thresholds are reproducible.

Channels (38): 3 IMUs × 8 (`ax ay az gx gy gz roll pitch`) + 14 encoder channels
(drive `pos/vel/eff`, swivel `pos`, 4 rollers × `pos/vel`, lift `pos/eff`).
Position channels are window-mean-subtracted — absolute position is not a fault signal.

### Input normalization

Inputs are standardized per site by `NormStats` (`fl/amr_schema.py`). The site
keeps accumulating channel statistics until every load state (empty / light /
heavy) has 20 windows, or round 5 arrives, and then **freezes** them. Normalizing
from round 1 alone blew the loss up whenever a later round brought a load state
the first round had not seen. z values are clipped to ±10.

| Variable | Default |
| --- | --- |
| `AMR_NORM_MIN_PER_LOAD` | 20 |
| `AMR_NORM_FREEZE_ROUND` | 5 |
| `AMR_NORM_CLIP` | 10 |

### Anomaly scoring

The default score (`FL_SCORE_MODE=diff`) compares the *change* between consecutive
samples, not the raw signal: per channel, `mean_t |Δx − Δx̂|`. All three fault
types show up as one-sample jolts on the accelerometers, and in the raw-signal
error those jolts are buried under the low-frequency error of reconstructing the
trajectory. The first difference keeps the jolt.

Each channel's error is turned into a robust z score against the reference for its
**condition cell** (navigation group × speed bin × load state): `(e − median) / MAD`,
with the cell's MAD floored at 10 % of the channel's overall MAD so a quiet cell
cannot blow small errors up. A cell with fewer than 8 reference windows falls back
to the channel's overall reference. The window score is the **maximum** over channels,
because a floor fault shows up on a few channels, not on the average of 38.
`FL_SCORE_MODE=recon` restores the older per-channel reconstruction error.

Threshold: `median + 8 · MAD · 1.4826` over the reference windows
(`FL_THRESHOLD_N_SIGMA`, 8 by default; 3 when `FL_COND_NORM=0`).

Not `mean + 3σ`. The reference windows come from the training stream, and the
training stream is contaminated, so the mean and the standard deviation are both
dragged upward by the very windows the threshold is meant to catch. The median and
the MAD barely move until contamination approaches half, and the 1.4826 factor
rescales the MAD onto the same footing as σ. The multiplier is 8 rather than 3
because conditioning concentrates the normal scores and shrinks the MAD with them;
`fl/config.py` keeps the measurements behind that choice.

### Verdict: score before training, against a frozen reference

A round's windows are scored by the **previous round's detector before the site
trains on them**. Scoring after training let a fault be absorbed: the model learned
the faulty windows first and then judged them normal (a run that scored 200–400
against a fixed reference scored 9–11 once absorbed, under a threshold of 15–17).

- **Frozen reference.** The reference pool (rounds 1–5) is frozen at
  `FL_REF_FREEZE_ROUND` (5). After that, references and thresholds come only from
  that pool, so later anomalies cannot move the bar.
- **Flagged windows are not trained on.** Once the pool is frozen, windows above
  the threshold are left out of that round's training (`FL_EXCLUDE_FLAGGED=1`),
  as long as at least 8 windows remain.
- **Verdict rule.** Three consecutive windows above the threshold from the same
  robot (`K=3`) produce one verdict per robot, from round `FL_VERDICT_START_ROUND`
  (6) onward. Earlier rounds are a warm-up and never alarm.

### Measuring a round against a fixed set

`val_loss` and `val_auroc` used to be computed on whatever windows that round
happened to receive. Round 3 and round 9 were measured with different rulers, so
the numbers could not be compared; and since the live stream carries no labels,
`val_auroc` was a single-class 0.5 every time.

`fl/valset.py` and `fl/valset_amr.npz` fix the ruler in place. The coordinator
scores every global model against the same labelled windows and publishes the
result to `cnt-fl-inference`, which the dashboard shows as a per-round log.
`FL_VALSET_PATH` points at a different set. What the shipped set contains, and why
that matters when reading the numbers, is in Known limitations.

Labels are used for **scoring only**. Normalization statistics and the threshold
are both derived from the validation set without consulting them, so the
measurement path is unsupervised end to end.

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

The verdict is written to `ANALYSIS-AE-i/cnt-verdict` on the site's own MN-CSE.
Dispatching the second robot and moving the faulty one are sandbox actions, not FL
actions.

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

This is AdvantEDGE's own control API, not an ETSI MEC service API — no GS MEC
number applies. It stands in for what a real network would do on its own.
`fl/mec_app.py` also reads its `applications` endpoint to look up the app
instance ID that MEC 011 needs. The instances themselves (`edgeway-fl-a/b/c`)
are defined by the `edgeway` scenario; the FL side never creates or deletes
them, the way a real MEC orchestrator would own them.

`fl/mec_sandbox.py` wraps these: degrade an AP, disconnect a robot, hand it over,
restore. MEC028 names APs by BSSID while the controller names them by element
name, so `fl/mec_topology.py` holds the translation table (regenerable from the
scenario YAML via `export_from_yaml()`).

### Scenario 2 — handover

Degrade the serving AP → detect through MEC028 → pick a target (the robot's own
neighbor report first, since that is evidence the robot can hear it; otherwise the
least-loaded AP in the same site; `spare` site last, because leaving the site
changes MEP and would need MEC021, ETSI GS MEC 021 Application Mobility Service,
which is deployed but not called here) → request MOBILITY → confirm the new AP.

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
| site-c | mep-c | 10 | 2 |
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
  edge_node.py               local training (trimmed loss), calibration, scoring
  aggregator.py              FedAvg
  onem2m_utils.py            oneM2M client, dual-CSE routing, label discovery
  onem2m_pch.py              <pollingChannel> receiver for firewalled AEs
  amr_ingest.py              CIN → training windows (per robot, per site)
  prepare_data_amr.py        offline JSONL → pkl
  setup_resources_standard.py   resource/ACP/subscription setup, AE verification
  in_ae_standard.py          IN-AE coordinator, per-round evaluation
  mn_ae_standard.py          site trainer (TRAINING-AE-i) + analysis app
  mec_topology.py            AP↔BSSID, robot AE↔UE↔MAC tables
  mec_wai.py                 MEC028 client and link judgement
  mec_sandbox.py             MEC Sandbox controller events
  net_scenarios.py           scenarios 2 and 3
  personalize.py             local fine-tuning for the non-contributing node
  valset.py                  fixed validation set, per-round model scoring
  valset_amr.npz             the validation windows themselves
  build_valset.py            builds valset_amr.npz from Isaac Sim captures
  verdict.py                 writes floor verdicts to the analysis app's container
  dashboard_server.py        dashboard backend
  check_telemetry.py         diagnostic: how full each robot's telemetry container is
  probe_cra.py               diagnostic: reproduces the trainer's incremental fetch
  probe_drop.py              diagnostic: CINs dropped while building windows
  demo_verdict_sub.py        diagnostic: verdict subscription path end to end

fl_bearing_dashboard.html    dashboard front end
FL-diagrams/                 PlantUML sequence/structure diagrams
out/FL-diagrams/             rendered diagrams
archive/femto/               previous FEMTO bearing demo, kept for reference
archive/legacy/              superseded entry points
env.local.example            environment template (copy to .env.local)
requirements.txt             Python dependencies
run_fl.sh                    start the whole demo
run_experiment1.sh           experiment 1: ten rounds end to end, no detection
clean_fl.sh                  reset resources and local artifacts
logs/                        per-run logs (git-ignored)
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
mep-a    https://<host>/<sandbox>/mep-a/meep-acme-mn-cse    acme-cse-mn-mep-a-<sandbox>
MEC028   https://<host>/<sandbox>/mep-a/wai/v2
control  https://<host>/<sandbox>/sandbox-ctrl/v1
```

**The IN-CSE is not one of these.** It is a shared CSE outside the sandbox, so
`FL_IN_BASE_URL` and `FL_IN_CSE_NAME` must be set explicitly in `.env.local`
(with `FL_IN_USER` / `FL_IN_PASS` if it asks for HTTP basic auth). The derived
sandbox-internal default exists only as a fallback and does not match a sandbox
deployment's actual path. `run_fl.sh` sources `.env.local` itself, so forgetting
to `source` it no longer sends every request to the wrong CSE.

### Federated learning

| Variable | Default | Meaning |
| --- | --- | --- |
| `FL_AGG_WEIGHTING` | `uniform` | `samples` for textbook sample-count FedAvg |
| `FL_SENSOR_INCREMENTAL` | `1` | `0` refetches the latest N instead of only new CINs |
| `FL_SENSOR_MAX_WINDOWS_PER_SITE` | `50` | per-round cap per site; `0` disables |
| `AMR_MAX_NORMAL_PER_SITE` | `0` | offline cap: `min`, a number, or `0` |
| `FL_ROBOT_WAIT_SEC` | `300` | how long to wait for Isaac Sim to register robots |
| `FL_MIN_ROBOTS_PER_SITE` | `1` | robots required per site before training starts |
| `FL_LOCAL_ONLY_NODES` | `mn3` | receives the global model, never contributes |
| `FL_COLD_START_NODES` | — | excluded from training entirely |
| `FL_SENSOR_FROM_CSE` | `1` | `0` falls back to local pkl (offline) |
| `FL_SENSOR_MIN_WINDOWS` | `50` | windows a site waits for before a round proceeds |
| `FL_SENSOR_BACKLOG_SEC` | `0` | `0` uses everything already in the containers; a positive value only what arrived within that many seconds; negative skips the backlog entirely |
| `FL_LABEL_FOR_TRAIN` | `0` | `1` trains on label-selected normals (supervised) |
| `FL_TRIM_RATIO` | `0` | fraction of each batch dropped by loss; `0` disables |
| `FL_TRIM_WARMUP_EPOCHS` | `2` | epochs before trimming starts |
| `FL_VERDICT` | `1` | `0` stops writing verdicts to the MN-CSE |
| `FL_VERDICT_START_ROUND` | `6` | first round allowed to post a verdict |
| `FL_REF_FREEZE_ROUND` | `5` | round at which the reference pool is frozen |
| `FL_EXCLUDE_FLAGGED` | `1` | leave above-threshold windows out of training |
| `FL_SCORE_MODE` | `diff` | `recon` for plain reconstruction error |
| `FL_THRESHOLD_N_SIGMA` | `8` | MAD multiplier of the threshold (`3` with `FL_COND_NORM=0`) |
| `FL_MIN_STEPS` | `200` | minimum optimizer steps per local round; epochs are raised to reach it |
| `FL_DP` | `0` | `1` turns per-batch clipping + noise back on (comparison only) |
| `FL_LBL_SITE_TELEMETRY` | `{site}:telemetry` | combined label used to discover telemetry |
| `FL_VALSET_PATH` | `fl/valset_amr.npz` | fixed validation set |
| `FL_SENSOR_CIN_LIMIT_PER_ROBOT` | `150` | CINs fetched per robot per round |
| `FL_MAX_CIN_BYTES` | 4 MB | embed ceiling before falling back |
| `FL_CHUNK_WEIGHTS` | `0` | fragment weights across CINs |
| `FL_CREATE_AES` | `0` | robot AEs come from Isaac Sim |

Model and training defaults: `seq_len=60`, `n_channels=38`, `cond_dim=11`,
`latent_dim=24`, `hidden=48`, `beta=0.5`; 10 global rounds, 10 local epochs
(raised as needed to reach `FL_MIN_STEPS`), batch 32, lr 1e-3. Differential
privacy is off by default: with per-batch clipping the added noise was 3–8× the
clipped gradient, and the global model converged to near-zero output.

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

1. The MEC Sandbox `edgeway` scenario is deployed — one MN-CSE per MEP.
2. The IN-CSE is reachable and `FL_IN_BASE_URL` / `FL_IN_CSE_NAME` point at it.
3. Isaac Sim has registered the robot AEs and is publishing telemetry.
4. `.env.local` is filled in (see Configuration). `run_fl.sh` reads it for you.

Start the simulator **before** the FL run if you can. Nothing is lost if you do
not — the trainers read whatever is already in the telemetry containers, oldest
first — but a round only advances once `FL_SENSOR_MIN_WINDOWS` windows exist.

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
./run_fl.sh                           # resources → dashboard → 3 trainers → IN-AE
FL_SENSOR_FROM_CSE=0 ./run_fl.sh      # offline: local pkl instead of the CSE
```

Or individually:

```bash
python3 fl/mn_ae_standard.py 0 5001   # TRAINING-AE-1  site A  mep-a
python3 fl/mn_ae_standard.py 1 5002   # TRAINING-AE-2  site B  mep-b
python3 fl/mn_ae_standard.py 2 5003   # TRAINING-AE-3  site C  mep-c
python3 fl/in_ae_standard.py          # coordinator
python3 fl/dashboard_server.py        # http://localhost:7000
```

Start the site trainers **before** the coordinator. Their `cnt-fl-control`
subscriptions must exist first; otherwise the first round command notifies nobody.

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
| TRAINING-AE-1 notification server | 5001 |
| TRAINING-AE-2 notification server | 5002 |
| TRAINING-AE-3 notification server | 5003 |
| IN-AE notification server | 6000 |
| Dashboard | 7000 |

`node_id` and port must stay aligned: `0 → mn1 → 5001`, `1 → mn2 → 5002`,
`2 → mn3 → 5003`.

---

## 11. Known limitations

Stated plainly, because the measurements are what they are.

**One live three-site run detected the injected fault, with caveats.** On
2026-11-19 (11:36 run, live Isaac Sim telemetry, ten rounds), a floor fault on
`siteC_AMR_2` starting around 11:45:15 was flagged in round 8 and the verdict was
posted at 11:46:11 (score 33.2, threshold 15.23); `siteC_AMR_1` raised none. The
same run also showed:

- **Two false alarms at site A** (rounds 6 and 10), both in low-speed turn cells
  (`to_pick/turn_in`, `to_pick/turn_start`) that the frozen reference pool barely
  covers.
- **Site A lags behind its robots.** Four robots produce about 1.3 windows/s while
  a round takes 50, so the backlog grew from 143 to 469 windows and site A was
  about 5.5 minutes behind by round 10.
- **The validation AUROC falls below the no-model baseline after round 2**
  (baseline 0.8323 in `diff` mode), so the per-round gain on the fixed set is not
  yet shown.

One run is not a measurement. Earlier two-node rehearsals predate the current
scoring and are not reproduced here.

**Site C is data-starved.** Only one robot has usable data, 25 moving windows.
The design calls for four. Normal driving at site A's scale (~400 windows) and
50+ bump samples are needed before the comparison means anything.

**Floor faults are more detectable than robot faults.** A single channel
(`roller_rl.vel`) reaches AUROC 0.855 on floor bumps versus about 0.71 for robot
faults — so scenario 5's floor side is on firmer ground than its robot side.

**The MEC028 thresholds in section 8 are estimates.** They have not been checked
against a running sandbox. If normal RSSI sits outside the assumed range,
scenario 2 will either miss the degradation or flag healthy links.

**Each site normalizes its inputs by its own statistics, and sharing them makes
things worse.** A global model carries no normalization statistics: every site
accumulates its own channel means and standard deviations and freezes them (see
Input normalization), so FedAvg is averaging weights learned in three differently
scaled input spaces. Pooling the statistics and giving every site the same ruler is the
obvious fix, and it was measured: on a two-node FedAvg rehearsal (five seeds, ten
rounds, two separate captures as the two sites) shared statistics scored AUROC
0.5772 against 0.6752 for per-site statistics, losing in all fifty comparisons.
Per-site normalization appears to be doing real work — each site's normal driving
lands near zero on its own scale — and flattening that costs more than the
mismatch does. The mechanism has not been measured, so the change is not made.

**The validation set is normalized by its own statistics, and that turns out to
be the right choice.** `fl/valset.py` standardizes the validation windows against
themselves rather than borrowing a site's ruler. Scoring the same global models
four ways — the validation set's own statistics, each site's, and the two pooled —
the validation set's own won every one of fifty comparisons: AUROC 0.6752 against
0.6248 (site 2), 0.5927 (pooled) and 0.5280 (site 1), with false alarms lowest as
well. An earlier single-node rehearsal suggested the opposite and was wrong.

**The validation set comes from one robot at one site.** `siteC_AMR_2`, a single
continuous run. It says nothing about how the model generalizes across sites,
which is the whole premise of the federated setup.

**The sandbox scenario YAML lists four robots at site B** while the code and data
use two (`ROBOTS_PER_NODE`). The YAML is being corrected.

**Earlier evaluation numbers came from a leaky split.** Until the run-aware split
landed, training and validation drew neighbouring one-second windows from the
same traversal, and the validation normals were reused inside the test set. Any
AUROC measured before that fix was optimistic, and with only two normal windows
on the honest side at site C it was also close to noise.

---

## 12. Standards referenced

| Standard | Use |
| --- | --- |
| oneM2M TS-0001 | resource types, ACP semantics |
| oneM2M TS-0004 | AE registration attributes (`rr`, `srv`, `api`) |
| oneM2M TS-0009 | HTTP binding, mandatory `X-M2M-RI` |
| oneM2M TR-0084 | federated learning over oneM2M resources |

### ETSI MEC

| Standard | Title | Sandbox service | In this repository |
| --- | --- | --- | --- |
| ETSI GS MEC 003 | Framework and Reference Architecture | — | terminology only (MEP, MEC host) |
| ETSI GS MEC 011 | Edge Platform Application Enablement | `mec011-N` | **called** — `fl/mec_app.py`: `confirm_ready` → `registrations` (app support, `isInsByMec=false`), `GET services` (service management) to find 013 and 028 |
| ETSI GS MEC 013 | Location API | `mec013-N` | **called** — `fl/mec_loc.py`, queries `users` / `zones`; zone and AP go into `cnt-net-status` and verdicts |
| ETSI GS MEC 021 | Application Mobility Service API | `mec021-N` | not called; would be needed for a cross-MEP handover to `spare` (section 5) |
| ETSI GS MEC 028 | WLAN Information API (WAI) | `mec028-N` | **called** — `fl/mec_wai.py`, queries `ap_information` / `sta_information`; scenarios 2 and 3, and once per round via `fl/mec_services.py` |
| ETSI GS MEC 033 | IoT API | `mec033-N` | **called** — `fl/mec_iot.py`: `GET registered_iot_platforms` gives each site app its MN-CSE address (checked with a CSEBase retrieve; falls back to the naming rule in `config.mn_endpoint`) |
| ETSI GS MEC 040 | Federation enablement APIs | `mec040-N` | not called |
| ETSI GS MEC 046 | Sensor-sharing API | `mec046-N` | not called |

`N` is the MEP index (1 = mep-a, 2 = mep-b, 3 = mep-c). The MEC Sandbox
controller (`sandbox-ctrl/v1/events/…`) is not in this table because it is not an
ETSI MEC specification.

CSE: [ACME](https://github.com/ankraft/ACME-oneM2M-CSE).
Network emulation: [AdvantEDGE / ETSI MEC Sandbox](https://github.com/InterDigitalInc/AdvantEDGE).
