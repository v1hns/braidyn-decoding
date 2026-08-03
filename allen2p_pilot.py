"""Feasibility pilot: does the pooling/drift result reproduce on Allen Visual Behavior 2-photon?

WHY THIS IS A PILOT AND NOT A REPLICATION. Cross-subject decoding needs feature correspondence
across animals. BraiDyn has it for free: parcel 3 is MOp in every mouse. Two-photon data does not --
mouse A's neuron #17 is unrelated to mouse B's neuron #17 -- and the longitudinal subset of this
cohort is dominated by SINGLE-PLANE recordings (114 sessions / 16 mice at VISp 150um), so averaging
by area gives a 1-dimensional feature and is useless.

The only representation comparable across animals with no cell correspondence is a
PERMUTATION-INVARIANT summary of the population response: statistics of the distribution of
per-neuron evoked responses, which do not depend on neuron identity or on how many neurons were
segmented. This pilot asks one question: do such features decode the task at all, and is there any
sign of the personal-vs-pooled aging asymmetry? If both are flat, the direction closes cheaply.

Design (mirrors pooling_leakfix.py so the numbers are comparable):
  cohort   : containers spanning >=14 days, all experiments VISp at 125-175um, one plane
  events   : trial change_time; label = is_change (go vs catch) -- a STIMULUS contrast, which is
             less reward/lick-coupled than the hit-vs-miss contrast would be
  feature  : per neuron, mean dF/F over 0..+1 s minus mean over -0.5..0 s, then the population
             distribution summarised by quantiles + mean + sd (permutation-invariant, fixed length)
  blocks   : each container's sessions split into an early half and a late half by date
  WS / WX  : leave-one-EARLY-SESSION-out; score on the held-out early session and on the late block
  LS / LX  : decoder trained on the OTHER mice's early sessions, scored on the same two
  asymmetry: (WS-WX) - (LS-LX), positive = the mouse's own decoder ages more

Streams NWB from S3; never downloads a whole file. Writes allen2p_pilot.json.
"""
import csv, datetime, json, os, socket, sys
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

socket.setdefaulttimeout(120)
HERE = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(HERE, "ophys_exp.csv")
S3 = ("https://visual-behavior-ophys-data.s3.us-west-2.amazonaws.com/"
      "visual-behavior-ophys/behavior_ophys_experiments/behavior_ophys_experiment_{}.nwb")
PRE, POST = 0.5, 1.0
QUANTS = np.linspace(0.02, 0.98, 25)     # 25 quantiles + mean + sd -> 27-dim, fixed across mice
N_MICE = int(sys.argv[1]) if len(sys.argv) > 1 else 8


def parse_dt(s):
    s = s.strip().replace("T", " ").split("+")[0].split(".")[0]
    for f in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(s[:19] if len(s) > 10 else s, f)
        except ValueError:
            pass
    return None


def depth_bin(x):
    return (int(float(x)) // 50) * 50


def cohort():
    """Mice whose >=14-day container sits entirely in VISp at one 50um depth bin.

    The pooled decoder needs every mouse in the same (area, depth) band, since that band IS the
    cross-subject correspondence here. The 150-199um group is the largest such band: 25 mice.
    """
    rows = list(csv.DictReader(open(META)))
    cont = defaultdict(list)
    for r in rows:
        dt = parse_dt(r["date_of_acquisition"])
        if dt:
            cont[r["ophys_container_id"]].append((dt, r))
    groups = defaultdict(list)
    for c, v in cont.items():
        span = (max(x[0] for x in v) - min(x[0] for x in v)).days
        rs = [x[1] for x in v]
        if span < 14 or not all(x["targeted_structure"] == "VISp" for x in rs):
            continue
        bins = {depth_bin(x["imaging_depth"]) for x in rs}
        if len(bins) != 1:
            continue
        groups[bins.pop()].append({"container": c, "mouse": rs[0]["mouse_id"], "span": span,
                                   "sessions": sorted((x[0], x[1]["ophys_experiment_id"]) for x in v)})
    band = max(groups, key=lambda b: len({o["mouse"] for o in groups[b]}))
    out = groups[band]
    best = {}
    for o in out:
        m = o["mouse"]
        if m not in best or len(o["sessions"]) > len(best[m]["sessions"]):
            best[m] = o
    sel = sorted(best.values(), key=lambda o: (-len(o["sessions"]), -o["span"]))[:N_MICE]
    print(f"depth band {band}-{band+49}um selected ({len(best)} mice available)", flush=True)
    return sel


MIN_CELLS = 15          # quantile features are meaningless on a handful of ROIs


def find_dff(f):
    """dF/F lives at processing/ophys/dff/traces (time x neurons), verified by probe."""
    g = f["processing"]["ophys"]["dff"]["traces"]
    return g["data"], np.asarray(g["timestamps"])


def build(args):
    mouse, day, eid = args
    import remfile, h5py
    try:
        f = h5py.File(remfile.File(S3.format(eid)), "r")
        ds, ts = find_dff(f)
        if ds is None or ts is None:
            return mouse, day, eid, None, "no dff"
        X = np.asarray(ds)
        if X.shape[0] == len(ts):        # stored (time, neurons) -> want (neurons, time)
            X = X.T
        if X.shape[0] < MIN_CELLS:
            return mouse, day, eid, None, f"only {X.shape[0]} cells"
        tr = f["intervals"]["trials"]
        ct = np.asarray(tr["change_time"])
        isch = np.asarray(tr["is_change"]).astype(bool)
        ab = np.asarray(tr["aborted"]).astype(bool) if "aborted" in tr else np.zeros(len(ct), bool)
        ok = (~ab) & np.isfinite(ct)
        ct, isch = ct[ok], isch[ok]
        if ok.sum() < 40 or isch.sum() < 10 or (~isch).sum() < 10:
            return mouse, day, eid, None, f"few trials ({ok.sum()})"
        rate = 1.0 / np.median(np.diff(ts[:500]))
        a, b = int(PRE * rate), int(POST * rate)
        feats, labs = [], []
        for t, lab in zip(ct, isch):
            i = int(np.searchsorted(ts, t))
            if i - a < 0 or i + b >= X.shape[1]:
                continue
            ev = X[:, i:i + b].mean(1) - X[:, i - a:i].mean(1)   # per-neuron evoked
            ev = ev[np.isfinite(ev)]
            if ev.size < 5:
                continue
            feats.append(np.r_[np.quantile(ev, QUANTS), ev.mean(), ev.std()])
            labs.append(int(lab))
        if len(labs) < 30 or len(set(labs)) < 2:
            return mouse, day, eid, None, "few usable"
        return mouse, day, eid, (np.array(feats), np.array(labs), X.shape[0]), "ok"
    except Exception as e:
        return mouse, day, eid, None, str(e)[:60]


def pipe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))


def auc(clf, X, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, clf.predict_proba(X)[:, 1]))


def main():
    sel = cohort()
    print(f"cohort: {len(sel)} mice (VISp 125-175um, single plane, container span >=14 d)", flush=True)
    for s in sel:
        print(f"  {s['mouse']:>10}  span {s['span']:2d} d  {len(s['sessions'])} sessions", flush=True)

    jobs = [(s["mouse"], d, e) for s in sel for d, e in s["sessions"]]
    print(f"\nstreaming {len(jobs)} experiments ...", flush=True)
    data = defaultdict(dict)
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(build, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            m, day, eid, res, msg = fut.result()
            if res:
                data[m][day] = res
            print(f"  [{i}/{len(jobs)}] {m} {day.date()} {msg}"
                  f"{'' if not res else f' n_neurons={res[2]} n_events={len(res[1])}'}", flush=True)

    mice = [m for m in data if len(data[m]) >= 4]
    print(f"\nusable mice (>=4 sessions): {len(mice)}", flush=True)
    if len(mice) < 4:
        print("TOO FEW MICE — pilot inconclusive\nPILOT_DONE"); return

    # early / late split within each mouse's container
    blocks = {}
    for m in mice:
        days = sorted(data[m])
        h = len(days) // 2
        e = [data[m][d] for d in days[:max(2, h)]]
        l = [data[m][d] for d in days[max(2, h):]]
        if not l:
            continue
        blocks[m] = (days[:max(2, h)], e,
                     (np.vstack([x[0] for x in l]), np.concatenate([x[1] for x in l])))
    mice = [m for m in mice if m in blocks]
    print(f"mice with early+late blocks: {len(mice)}", flush=True)

    rows = []
    for m in mice:
        edays, elist, (Xl, yl) = blocks[m]
        others = [o for o in mice if o != m]
        Xo = np.vstack([x[0] for o in others for x in blocks[o][1]])
        yo = np.concatenate([x[1] for o in others for x in blocks[o][1]])
        if len(set(yo)) < 2 or len(set(yl)) < 2:
            continue
        clf_pool = pipe().fit(Xo, yo)
        ws, wx, ls = [], [], []
        for i in range(len(elist)):
            tr = [elist[j] for j in range(len(elist)) if j != i]
            if not tr:
                continue
            Xt = np.vstack([x[0] for x in tr]); yt = np.concatenate([x[1] for x in tr])
            Xh, yh = elist[i][0], elist[i][1]
            if len(set(yt)) < 2 or len(set(yh)) < 2:
                continue
            c = pipe().fit(Xt, yt)
            ws.append(auc(c, Xh, yh)); wx.append(auc(c, Xl, yl))
            ls.append(auc(clf_pool, Xh, yh))
        if not ws:
            continue
        WS, WX, LS = float(np.mean(ws)), float(np.mean(wx)), float(np.mean(ls))
        LX = auc(clf_pool, Xl, yl)
        rows.append({"mouse": m, "n_sessions": len(data[m]), "WS": WS, "WX": WX, "LS": LS, "LX": LX,
                     "personal_aging": WS - WX, "pooled_aging": LS - LX,
                     "asymmetry": (WS - WX) - (LS - LX)})
        print(f"  {m:>10}  WS {WS:.3f} WX {WX:.3f} | LS {LS:.3f} LX {LX:.3f} | "
              f"asym {(WS-WX)-(LS-LX):+.4f}", flush=True)

    if not rows:
        print("no usable rows\nPILOT_DONE"); return
    a = np.array([r["asymmetry"] for r in rows])
    rng = np.random.default_rng(1)
    null = np.array([(a * rng.choice([-1, 1], len(a))).mean() for _ in range(20000)])
    p = float((1 + np.sum(np.abs(null) >= abs(a.mean()))) / (1 + len(null)))
    out = {"n_mice": len(rows), "per_mouse": rows,
           "within_early_AUC": float(np.mean([r["WS"] for r in rows])),
           "pooled_early_AUC": float(np.mean([r["LS"] for r in rows])),
           "personal_aging": float(np.mean([r["personal_aging"] for r in rows])),
           "pooled_aging": float(np.mean([r["pooled_aging"] for r in rows])),
           "asymmetry": float(a.mean()), "p_signflip": p,
           "frac_positive": float(np.mean(a > 0))}
    print(f"\n=== PILOT RESULT (n={len(rows)} mice) ===")
    print(f"  decodability: within-mouse early AUC {out['within_early_AUC']:.3f} | "
          f"pooled (other mice) AUC {out['pooled_early_AUC']:.3f}")
    print(f"  personal aging {out['personal_aging']:+.4f}   pooled aging {out['pooled_aging']:+.4f}")
    print(f"  ASYMMETRY {out['asymmetry']:+.4f}  p={p:.4f}  ({out['frac_positive']:.0%} of mice)")
    print(f"  BraiDyn for comparison: asymmetry +0.017, p=5e-5")
    json.dump(out, open(os.path.join(HERE, "allen2p_pilot.json"), "w"), indent=1)
    print("\nsaved allen2p_pilot.json\nPILOT_DONE", flush=True)


if __name__ == "__main__":
    main()
