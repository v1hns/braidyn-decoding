"""BraiDyn-BC cross-SESSION drift study (DANDI:001425).

The full study showed the cortex-wide behavioral code is largely conserved across individuals,
but used ONE session per mouse. The dataset is a ~3-week longitudinal protocol: every mouse has
~13-15 operant TASK sessions (day1..day15). This asks the obvious next question: is the conserved
code also STABLE over time, or does it drift across days?

ALL FOUR TARGETS (2026-07-16). This script originally ran lever-pull + lick only. That was NOT a
trial-count limitation -- it predated braidyn_4targets.py, which showed reward/tone were only ever
"eventless" because the old main analysis read sorted(paths)[0], often a RESTING-STATE session. On
task sessions reward/tone are well populated, so the early/late split is run for all four here.

Design (targets: `state_lever`, `lick`, `reward`, `tone`):
  Per mouse, split its task sessions into EARLY (day<=5) and LATE (day>=11) blocks.
  Event-triggered features (44 Allen parcels) exactly as the full study, per session, pooled per block.

  2x2 headline (AUC per held-out mouse, averaged over mice):
    WS  within-mouse same-block   : 5-fold CV on EARLY (baseline within-session number)
    WX  within-mouse cross-block  : train EARLY, test LATE, SAME mouse         -> within-animal drift
    LS  LOMO same-block           : train EARLY of other mice, test EARLY held-out -> conservation (replicates full study)
    LX  LOMO cross-block          : train EARLY of other mice, test LATE  held-out -> conserved AND stable?
  Cluster-bootstrap 95% CI over mice for each; paired mouse-level test LX vs LS (is there a temporal gap?).

  Drift curve: within-mouse, train on task session i, test on session j (j>i); AUC binned by day-gap.
Streams only parcellated dF/F + behavior via remfile. Saves braidyn_drift.json.
"""
import re, json, sys, os, socket
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

socket.setdefaulttimeout(90)   # fail a hung range-read rather than block the whole map forever

TARGETS = ["state_lever", "lick", "reward", "tone"]
EARLY_MAX = 5      # EARLY block = task days 1..5
LATE_MIN = 11      # LATE block  = task days 11..15


# ---------- data access (identical logic to full study) ----------
def list_task_sessions():
    from dandi.dandiapi import DandiAPIClient
    with DandiAPIClient() as c:
        d = c.get_dandiset("001425", "draft")
        beh = [a.path for a in d.get_assets()
               if a.path.endswith(".nwb") and "behavior" in a.path and "task" in a.path]
    by = defaultdict(list)
    for p in beh:
        mouse = re.search(r"(sub-[^/]+?)/", p).group(1)
        day = int(re.search(r"day(\d+)", p).group(1))
        by[mouse].append((day, p))
    return {m: sorted(v) for m, v in by.items()}   # mouse -> [(day, path), ...] task only


def stream(path):
    import remfile, h5py, pynwb
    from dandi.dandiapi import DandiAPIClient
    with DandiAPIClient() as c:
        d = c.get_dandiset("001425", "draft")
        url = d.get_asset_by_path(path).get_content_url(follow_redirects=1, strip_query=True)
    f = h5py.File(remfile.File(url), "r")
    nwb = pynwb.NWBHDF5IO(file=f, load_namespaces=True).read()
    dff = np.asarray(nwb.processing["ophys"]["DfOverF"]["dFF"].data)
    dn = nwb.processing["downsampled"]
    beh = {}
    for k in TARGETS:
        try:
            beh[k] = np.asarray(dn[k].data).ravel()
        except Exception:
            beh[k] = None
    ts = nwb.processing["ophys"]["DfOverF"]["dFF"].timestamps
    rate = 1.0 / np.median(np.diff(np.asarray(ts[:200]))) if ts is not None else 30.0
    return dff, beh, rate


def onsets(sig, rate):
    if sig is None:
        return np.array([], int)
    x = sig.astype(float)
    if np.nanmax(x) <= np.nanmin(x):
        return np.array([], int)
    b = (x > (np.nanmax(x) + np.nanmin(x)) / 2).astype(int)
    on = np.where(np.diff(b) == 1)[0] + 1
    keep = []
    for o in on:
        if not keep or o - keep[-1] > rate:
            keep.append(o)
    return np.array(keep, int)


def feats(dff, ev, rate, pre=0.5, post=1.0):
    T = dff.shape[0]; a = int(pre * rate); b = int(post * rate)
    out = [dff[e:e+b].mean(0) - dff[e-a:e].mean(0) for e in ev if e-a >= 0 and e+b < T]
    return np.array(out)


def build_session(args):
    """Return mouse, day, {target: (X, y)} for one task session (pos=event, neg=far baseline)."""
    mouse, day, path = args
    try:
        dff, beh, rate = stream(path)
    except Exception as e:
        return mouse, day, None, str(e)[:50]
    rng = np.random.default_rng(day)   # session-specific but deterministic
    T = dff.shape[0]; per = {}
    for tgt in TARGETS:
        ev = onsets(beh.get(tgt), rate)
        if len(ev) < 8:
            continue
        pos = feats(dff, ev, rate)
        far = [t for t in rng.integers(int(2*rate), T-int(2*rate), size=len(ev)*3)
               if np.min(np.abs(ev - t)) > 2*rate][:len(pos)]
        neg = feats(dff, np.array(far), rate)
        n = min(len(pos), len(neg))
        if n >= 6:
            per[tgt] = (np.vstack([pos[:n], neg[:n]]), np.r_[np.ones(n), np.zeros(n)])
    return mouse, day, per, f"rate {rate:.0f}, {list(per.keys())}, ev-days ok"


# ---------- decoding ----------
def pipe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))


def _pool(sessions, days):
    """Concatenate (X,y) over the sessions whose day is in `days`. Returns None if empty."""
    xs = [sessions[d][0] for d in days if d in sessions]
    ys = [sessions[d][1] for d in days if d in sessions]
    if not xs:
        return None
    return np.vstack(xs), np.concatenate(ys)


def analyze_drift(target, per_mouse_sessions):
    """per_mouse_sessions: mouse -> {day: (X,y)} for this target (task sessions only)."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import roc_auc_score

    mice = sorted(per_mouse_sessions)
    early = {}; late = {}
    for m in mice:
        sess = per_mouse_sessions[m]
        e = _pool(sess, [d for d in sess if d <= EARLY_MAX])
        l = _pool(sess, [d for d in sess if d >= LATE_MIN])
        if e is not None and l is not None and len(set(e[1])) == 2 and len(set(l[1])) == 2:
            early[m] = e; late[m] = l
    mice = sorted(early)   # mice with BOTH blocks usable

    def both_classes(y):
        return len(set(y)) == 2

    WS, WX, LS, LX = {}, {}, {}, {}
    for m in mice:
        Xe, ye = early[m]; Xl, yl = late[m]
        # WS within-mouse same-block CV on EARLY
        if len(ye) >= 12 and both_classes(ye):
            WS[m] = float(cross_val_score(pipe(), Xe, ye,
                          cv=StratifiedKFold(5, shuffle=True, random_state=0),
                          scoring="roc_auc").mean())
        # WX within-mouse cross-block: train EARLY -> test LATE (same mouse)
        if both_classes(ye) and both_classes(yl):
            clf = pipe(); clf.fit(Xe, ye)
            WX[m] = float(roc_auc_score(yl, clf.predict_proba(Xl)[:, 1]))
        # LOMO: train on EARLY of OTHER mice
        others = [o for o in mice if o != m]
        if others:
            Xtr = np.vstack([early[o][0] for o in others])
            ytr = np.concatenate([early[o][1] for o in others])
            if both_classes(ytr):
                clf = pipe(); clf.fit(Xtr, ytr)
                if both_classes(ye):   # LS test EARLY held-out
                    LS[m] = float(roc_auc_score(ye, clf.predict_proba(Xe)[:, 1]))
                if both_classes(yl):   # LX test LATE held-out
                    LX[m] = float(roc_auc_score(yl, clf.predict_proba(Xl)[:, 1]))

    def summ(d):
        v = np.array(list(d.values()))
        if len(v) == 0:
            return {"auc": None, "n": 0, "ci95": [None, None]}
        rng = np.random.default_rng(0)
        bs = [np.mean(rng.choice(v, len(v), replace=True)) for _ in range(2000)]
        return {"auc": float(v.mean()), "n": int(len(v)),
                "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}

    # paired LX vs LS over mice present in both (does the conserved code lose accuracy across time?)
    paired = [(LS[m], LX[m]) for m in mice if m in LS and m in LX]
    if paired:
        a = np.array([p[0] for p in paired]); b = np.array([p[1] for p in paired])
        diff = b - a   # LX - LS ; ~0 => no temporal gap
        rng = np.random.default_rng(1)
        # sign-flip permutation on paired differences
        null = np.array([np.mean(diff * rng.choice([-1, 1], len(diff))) for _ in range(5000)])
        p_gap = float((1 + np.sum(np.abs(null) >= abs(diff.mean()))) / (1 + len(null)))
        paired_stats = {"n_pairs": len(paired), "mean_LX_minus_LS": float(diff.mean()),
                        "p_two_sided": p_gap}
    else:
        paired_stats = None

    return {"target": target, "n_mice": len(mice),
            "WS_within_same": summ(WS), "WX_within_cross": summ(WX),
            "LS_lomo_same": summ(LS), "LX_lomo_cross": summ(LX),
            "paired_LX_vs_LS": paired_stats,
            "per_mouse": {m: {"WS": round(WS.get(m, float('nan')), 3),
                              "WX": round(WX.get(m, float('nan')), 3),
                              "LS": round(LS.get(m, float('nan')), 3),
                              "LX": round(LX.get(m, float('nan')), 3)} for m in mice}}


def drift_curve(target, per_mouse_sessions):
    """Within-mouse train-day-i test-day-j (j>i). AUC vs day-gap. Returns list of (gap, auc)."""
    from sklearn.metrics import roc_auc_score
    pts = []
    for m, sess in per_mouse_sessions.items():
        days = sorted(sess)
        for i in days:
            Xi, yi = sess[i]
            if len(set(yi)) != 2 or len(yi) < 12:
                continue
            clf = pipe(); clf.fit(Xi, yi)
            for j in days:
                if j <= i:
                    continue
                Xj, yj = sess[j]
                if len(set(yj)) != 2:
                    continue
                pts.append((j - i, float(roc_auc_score(yj, clf.predict_proba(Xj)[:, 1]))))
    # bin by gap
    binned = defaultdict(list)
    for g, a in pts:
        binned[g].append(a)
    curve = {int(g): {"auc": float(np.mean(v)), "n": len(v)} for g, v in sorted(binned.items())}
    # linear slope of AUC vs gap (drift rate per day)
    if len(pts) >= 10:
        g = np.array([p[0] for p in pts]); a = np.array([p[1] for p in pts])
        slope = float(np.polyfit(g, a, 1)[0])
    else:
        slope = None
    return {"curve": curve, "slope_per_day": slope, "n_pairs": len(pts)}


def main():
    task = list_task_sessions()
    jobs = [(m, day, path) for m, sess in task.items() for (day, path) in sess]
    n = len(jobs)
    print(f"mice: {len(task)}  task-sessions: {n}", flush=True)

    # target -> mouse -> {day: (X,y)}. Stream with high concurrency (network-latency bound),
    # collect via as_completed so progress is visible and one slow stream can't hide the rest.
    data = {t: defaultdict(dict) for t in TARGETS}
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=24) as ex:
        futs = {ex.submit(build_session, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                mouse, day, per, msg = fut.result()
            except Exception as e:
                fail += 1; print(f"  [{i}/{n}] ERR {str(e)[:50]}", flush=True); continue
            if per:
                ok += 1
                for tgt, xy in per.items():
                    data[tgt][mouse][day] = xy
            else:
                fail += 1
            if i % 25 == 0 or i == n:
                print(f"  streamed {i}/{n} (usable={ok}, skipped={fail})", flush=True)
    print(f"sessions with usable features: {ok}/{n}", flush=True)

    out = {"targets": {}}
    for tgt in TARGETS:
        pms = {m: dict(v) for m, v in data[tgt].items() if len(v) >= 2}
        print(f"\n=== {tgt}: {len(pms)} mice with >=2 task sessions ===")
        drift = analyze_drift(tgt, pms)
        curve = drift_curve(tgt, pms)
        out["targets"][tgt] = {"drift_2x2": drift, "drift_curve": curve}
        d = drift
        def g(k):
            x = d[k]["auc"]; return f"{x:.3f}" if x is not None else "  -  "
        print(f"  WS {g('WS_within_same')} | WX {g('WX_within_cross')} | "
              f"LS {g('LS_lomo_same')} | LX {g('LX_lomo_cross')}  (n_mice={d['n_mice']})")
        if d["paired_LX_vs_LS"]:
            ps = d["paired_LX_vs_LS"]
            print(f"  LX-LS = {ps['mean_LX_minus_LS']:+.3f}  p={ps['p_two_sided']:.3f}  (n={ps['n_pairs']})")
        print(f"  drift slope/day = {curve['slope_per_day']}  ({curve['n_pairs']} session-pairs)")

    # Write next to this script so the run works on any box, not just /home/ubuntu.
    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "braidyn_drift.json")
    json.dump(out, open(dest, "w"), indent=1)
    print(f"\nsaved {dest}")
    print("BRAIDYN_DRIFT_DONE")


if __name__ == "__main__":
    main()
