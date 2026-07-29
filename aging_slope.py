"""Continuous aging-RATE analysis: AUC-vs-day slope, personal vs pooled decoders (DANDI:001425).

The paper currently measures aging with a coarse block split (early days 1-5 vs late days 11-15).
This regresses decoding AUC against test-session DAY to get an aging *rate* (AUC/day) for:
  personal : decoder trained on the held-out mouse's OWN early sessions (day<=5)
  pooled   : decoder trained on the OTHER mice's early sessions (day<=5), leave-one-mouse-out
Both are tested on the held-out mouse's LATER sessions (day>5 -> no leakage for personal), and we
fit a per-mouse linear slope of AUC vs day for each. A paired sign-flip test asks whether the
personal decoder ages FASTER (steeper negative slope) than the pooled one. Also emits per-day mean
AUC curves (personal & pooled) for a decay figure. Saves aging_slope.json. Streams parcellated dF/F
only (remfile); no raw movie download.
"""
import re, json, os, socket
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

socket.setdefaulttimeout(90)
TARGETS = ["state_lever", "lick", "reward", "tone"]
EARLY_MAX = 5           # training block = task days 1..5
TEST_MIN_DAY = 6        # test only on day>5 sessions (no leakage for the personal decoder)
MIN_TEST_SESSIONS = 3   # need >=3 test days to fit a per-mouse slope
HERE = os.path.dirname(os.path.abspath(__file__))


# ---------- data access (identical logic to pooling_drift.py) ----------
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
    return {m: sorted(v) for m, v in by.items()}


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
    mouse, day, path = args
    try:
        dff, beh, rate = stream(path)
    except Exception as e:
        return mouse, day, None, str(e)[:50]
    rng = np.random.default_rng(day)
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
    return mouse, day, per, f"ok {list(per.keys())}"


# ---------- decoding ----------
def pipe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))


def _both(y):
    return len(set(y.tolist())) == 2


def _pool_days(sess, days):
    xs = [sess[d][0] for d in days if d in sess]
    ys = [sess[d][1] for d in days if d in sess]
    if not xs:
        return None
    return np.vstack(xs), np.concatenate(ys)


def analyze(target, per_mouse):
    from sklearn.metrics import roc_auc_score
    mice = sorted(per_mouse)
    early = {}
    for m in mice:
        e = _pool_days(per_mouse[m], [d for d in per_mouse[m] if d <= EARLY_MAX])
        if e is not None and _both(e[1]):
            early[m] = e
    mice = sorted(early)

    rows = []                       # (mouse, slope_personal, slope_pooled, n_test)
    curve_p = defaultdict(list)     # day -> [personal AUC]
    curve_l = defaultdict(list)     # day -> [pooled AUC]
    for m in mice:
        Xe, ye = early[m]
        others = [o for o in mice if o != m]
        Xtr = np.vstack([early[o][0] for o in others]); ytr = np.concatenate([early[o][1] for o in others])
        if not (_both(ye) and _both(ytr)):
            continue
        clf_p = pipe().fit(Xe, ye)          # personal
        clf_l = pipe().fit(Xtr, ytr)        # pooled (LOMO)
        days, ap, al = [], [], []
        for d in sorted(per_mouse[m]):
            if d < TEST_MIN_DAY:
                continue
            Xd, yd = per_mouse[m][d]
            if not _both(yd):
                continue
            a_p = float(roc_auc_score(yd, clf_p.predict_proba(Xd)[:, 1]))
            a_l = float(roc_auc_score(yd, clf_l.predict_proba(Xd)[:, 1]))
            days.append(d); ap.append(a_p); al.append(a_l)
            curve_p[d].append(a_p); curve_l[d].append(a_l)
        if len(days) >= MIN_TEST_SESSIONS:
            sp = float(np.polyfit(days, ap, 1)[0])   # AUC/day, personal
            sl = float(np.polyfit(days, al, 1)[0])   # AUC/day, pooled
            rows.append((m, sp, sl, len(days)))

    arr = np.array([(r[1], r[2]) for r in rows])
    sp_mean = float(arr[:, 0].mean()); sl_mean = float(arr[:, 1].mean())
    diff = arr[:, 0] - arr[:, 1]                      # personal - pooled ; <0 => personal ages faster
    rng = np.random.default_rng(0)
    null = np.array([(diff * rng.choice([-1, 1], len(diff))).mean() for _ in range(20000)])
    p = float((1 + np.sum(np.abs(null) >= abs(diff.mean()))) / (1 + len(null)))
    curve = {int(d): {"personal": float(np.mean(curve_p[d])), "pooled": float(np.mean(curve_l[d])),
                      "n": len(curve_p[d])} for d in sorted(curve_p)}
    return {"target": target, "n_mice": len(rows),
            "slope_personal_per_day": sp_mean, "slope_pooled_per_day": sl_mean,
            "mean_slope_diff_personal_minus_pooled": float(diff.mean()),
            "frac_personal_steeper": float(np.mean(diff < 0)), "p_signflip": p,
            "per_mouse": {r[0]: {"slope_personal": round(r[1], 5), "slope_pooled": round(r[2], 5),
                                 "n_test": r[3]} for r in rows},
            "decay_curve": curve}


def main():
    task = list_task_sessions()
    jobs = [(m, day, path) for m, sess in task.items() for (day, path) in sess]
    print(f"mice {len(task)}  task-sessions {len(jobs)}", flush=True)
    data = {t: defaultdict(dict) for t in TARGETS}
    ok = 0
    with ThreadPoolExecutor(max_workers=24) as ex:
        futs = {ex.submit(build_session, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            mouse, day, per, msg = fut.result()
            if per:
                ok += 1
                for t, xy in per.items():
                    data[t][mouse][day] = xy
            if i % 25 == 0 or i == len(jobs):
                print(f"  streamed {i}/{len(jobs)} usable={ok}", flush=True)

    out = {"targets": {}}
    for tgt in TARGETS:
        pm = {m: dict(v) for m, v in data[tgt].items() if len(v) >= 2}
        r = analyze(tgt, pm)
        out["targets"][tgt] = r
        print(f"  [{tgt}] personal {r['slope_personal_per_day']:+.5f}/day  "
              f"pooled {r['slope_pooled_per_day']:+.5f}/day  "
              f"diff {r['mean_slope_diff_personal_minus_pooled']:+.5f} p={r['p_signflip']:.4f} "
              f"(n={r['n_mice']}, personal steeper in {r['frac_personal_steeper']*100:.0f}%)", flush=True)

    dest = os.path.join(HERE, "aging_slope.json")
    json.dump(out, open(dest, "w"), indent=1)
    print(f"\nsaved {dest}\nAGING_SLOPE_DONE", flush=True)


if __name__ == "__main__":
    main()
