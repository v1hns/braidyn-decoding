"""BraiDyn-BC main conservation result for ALL FOUR targets (DANDI:001425).

The original main analysis dropped reward and tone for "lacking threshold-detectable events" --
an artifact of selecting sorted(paths)[0], which for many mice is a RESTING-STATE session with
no task events. On the operant TASK sessions, reward and tone are well-populated binary channels
(reward fires on each rewarded trial, tone on each trial cue). This recomputes the leave-one-
mouse-out conservation result for lever-pull, lick, reward, AND tone under ONE consistent
pipeline: evoked-difference features from task sessions (3/mouse spread across the protocol),
linear logistic decoder, within-mouse control, LOMO per-mouse AUC, cluster-bootstrap CI,
permutation null, and per-parcel importance. Output braidyn_4targets.json (drop-in for figs 1-3).
"""
import re, json, socket
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

socket.setdefaulttimeout(90)
TARGETS = ["state_lever", "lick", "reward", "tone"]
A_FR, B_FR = 15, 30          # fixed pre/post frames (~30 Hz)


def pick_sessions(k=3):
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
    out = {}
    for mouse, v in by.items():
        v = sorted(v); n = len(v)
        idx = sorted(set([n // 4, n // 2, (3 * n) // 4]))[:k] if n >= k else list(range(n))
        out[mouse] = [v[i] for i in idx]
    return out


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


def feats(dff, ev, a, b):
    T = dff.shape[0]
    out = [dff[e:e + b].mean(0) - dff[e - a:e].mean(0) for e in ev if e - a >= 0 and e + b < T]
    return np.array(out)


def build_session(args):
    mouse, day, path = args
    try:
        dff, beh, rate = stream(path)
    except Exception as e:
        return mouse, day, None, str(e)[:50]
    a, b = A_FR, B_FR
    rng = np.random.default_rng(day); T = dff.shape[0]; per = {}
    for tgt in TARGETS:
        ev = onsets(beh.get(tgt), rate)
        if len(ev) < 8:
            continue
        pos = feats(dff, ev, a, b)
        if len(pos) == 0:
            continue
        far = [t for t in rng.integers(a + 1, T - b - 1, size=len(ev) * 3)
               if np.min(np.abs(ev - t)) > 2 * rate][:len(pos)]
        neg = feats(dff, np.array(far), a, b)
        n = min(len(pos), len(neg))
        if n >= 6:
            per[tgt] = (np.vstack([pos[:n], neg[:n]]), np.r_[np.ones(n), np.zeros(n)])
    return mouse, day, per, f"rate {rate:.0f} {list(per.keys())}"


def analyze(target, data):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_score, LeaveOneGroupOut
    from sklearn.metrics import roc_auc_score
    mice = sorted(data)
    X = np.vstack([data[m][0] for m in mice])
    y = np.concatenate([data[m][1] for m in mice])
    g = np.concatenate([[m] * len(data[m][1]) for m in mice])
    pipe = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    wm = []
    for m in mice:
        idx = g == m
        if idx.sum() >= 12 and len(set(y[idx])) == 2:
            wm.append(cross_val_score(pipe(), X[idx], y[idx],
                      cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc").mean())
    yp = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        mm = pipe(); mm.fit(X[tr], y[tr]); yp[te] = mm.predict_proba(X[te])[:, 1]
    per_mouse = {m: roc_auc_score(y[g == m], yp[g == m]) for m in mice if len(set(y[g == m])) == 2}
    lomo = float(np.mean(list(per_mouse.values())))
    rng = np.random.default_rng(0); vals = np.array(list(per_mouse.values())); bs = []
    for _ in range(2000):
        bs.append(np.mean(rng.choice(vals, len(vals), replace=True)))
    ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    rng2 = np.random.default_rng(1); null = []
    for _ in range(150):
        yperm = y.copy()
        for m in mice:
            idx = np.where(g == m)[0]; yperm[idx] = rng2.permutation(yperm[idx])
        ypn = np.zeros(len(y))
        for tr, te in LeaveOneGroupOut().split(X, yperm, g):
            mm = pipe(); mm.fit(X[tr], yperm[tr]); ypn[te] = mm.predict_proba(X[te])[:, 1]
        null.append(np.mean([roc_auc_score(yperm[g == m], ypn[g == m])
                             for m in mice if len(set(yperm[g == m])) == 2]))
    null = np.array(null); pval = float((1 + np.sum(null >= lomo)) / (1 + len(null)))
    clf = pipe(); clf.fit(X, y)
    coef = np.abs(clf.named_steps["logisticregression"].coef_.ravel())
    top = np.argsort(-coef)[:8].tolist()
    return {"target": target, "n_mice": len(mice), "n_events": int(len(y) // 2),
            "within_mouse_auc": float(np.mean(wm)), "within_mouse_sd": float(np.std(wm)),
            "lomo_auc": lomo, "lomo_ci95": ci, "perm_p": pval, "perm_null_mean": float(np.mean(null)),
            "per_mouse_auc": {m: round(v, 3) for m, v in per_mouse.items()},
            "top_parcels": top, "parcel_importance": coef.round(3).tolist()}


def main():
    sess = pick_sessions()
    jobs = [(m, day, path) for m, v in sess.items() for (day, path) in v]
    n = len(jobs)
    print(f"mice {len(sess)}  task-sessions {n}", flush=True)
    data = {t: defaultdict(list) for t in TARGETS}
    ok = 0
    with ThreadPoolExecutor(max_workers=24) as ex:
        futs = {ex.submit(build_session, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            mouse, day, per, msg = fut.result()
            if per:
                ok += 1
                for tgt, xy in per.items():
                    data[tgt][mouse].append(xy)
            if i % 15 == 0 or i == n:
                print(f"  streamed {i}/{n} (usable={ok})", flush=True)
    # pool sessions per mouse
    pooled = {t: {} for t in TARGETS}
    for t in TARGETS:
        for m, lst in data[t].items():
            pooled[t][m] = (np.vstack([x for x, _ in lst]), np.concatenate([y for _, y in lst]))
    out = {}
    for tgt in TARGETS:
        d = {m: xy for m, xy in pooled[tgt].items() if len(set(xy[1])) == 2 and len(xy[1]) >= 12}
        print(f"\n=== {tgt}: {len(d)} mice ===", flush=True)
        r = analyze(tgt, d); out[tgt] = r
        print(f"  within {r['within_mouse_auc']:.3f}+/-{r['within_mouse_sd']:.3f} | "
              f"LOMO {r['lomo_auc']:.3f} CI{[round(c,3) for c in r['lomo_ci95']]} | "
              f"p={r['perm_p']:.3f} | {r['n_mice']} mice, {r['n_events']} events", flush=True)
    json.dump(out, open("/home/ubuntu/braidyn_4targets.json", "w"), indent=1)
    print("\nsaved braidyn_4targets.json\nBRAIDYN_4T_DONE", flush=True)


if __name__ == "__main__":
    main()
