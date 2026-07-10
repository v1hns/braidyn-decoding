"""BraiDyn-BC full study: cross-mouse cortex-wide decoding (DANDI:001425).

All 25 mice, 4 decode targets (lever-pull, reward, cue-tone, lick). For each target:
  - within-mouse positive control (5-fold, per mouse)
  - LEAVE-ONE-MOUSE-OUT (the novel virgin axis), per-mouse held-out AUC
  - cluster-bootstrap 95% CI (resample mice over their held-out AUCs)
  - permutation null (shuffle labels within mouse, redo LOMO)
  - per-parcel importance (standardized logistic |coef|, mean over folds)
Streams only the parcellated dF/F (44 Allen parcels) + behavior via remfile. Saves results.json.
"""
import re, json
import numpy as np
from concurrent.futures import ThreadPoolExecutor

TARGETS = ["state_lever", "reward", "tone", "lick"]


def pick_sessions():
    from dandi.dandiapi import DandiAPIClient
    with DandiAPIClient() as c:
        d = c.get_dandiset("001425", "draft")
        assets = [a.path for a in d.get_assets() if a.path.endswith(".nwb") and "behavior" in a.path]
    by = {}
    for p in assets:
        m = re.search(r"(sub-[^/]+?)/", p)
        by.setdefault(m.group(1), []).append(p)
    return {mouse: sorted(paths)[0] for mouse, paths in by.items()}   # 1 behavior session/mouse


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


def build_mouse(args):
    mouse, path = args
    try:
        dff, beh, rate = stream(path)
    except Exception as e:
        return mouse, None, str(e)[:60]
    per = {}
    rng = np.random.default_rng(0); T = dff.shape[0]
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
    return mouse, per, f"rate {rate:.0f}, targets {list(per.keys())}"


def analyze(target, data):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_score, LeaveOneGroupOut
    from sklearn.metrics import roc_auc_score
    mice = sorted(data)
    X = np.vstack([data[m][0] for m in mice])
    y = np.concatenate([data[m][1] for m in mice])
    g = np.concatenate([[m]*len(data[m][1]) for m in mice])
    pipe = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    # within-mouse control
    wm = []
    for m in mice:
        idx = g == m
        if idx.sum() >= 12 and len(set(y[idx])) == 2:
            wm.append(cross_val_score(pipe(), X[idx], y[idx], cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc").mean())
    # LOMO
    yp = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        mm = pipe(); mm.fit(X[tr], y[tr]); yp[te] = mm.predict_proba(X[te])[:, 1]
    per_mouse = {m: roc_auc_score(y[g==m], yp[g==m]) for m in mice if len(set(y[g==m])) == 2}
    lomo = float(np.mean(list(per_mouse.values())))
    # cluster bootstrap CI over mice
    rng = np.random.default_rng(0); vals = np.array(list(per_mouse.values())); bs = []
    for _ in range(2000):
        bs.append(np.mean(rng.choice(vals, len(vals), replace=True)))
    ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    # permutation null
    rng2 = np.random.default_rng(1); null = []
    for _ in range(150):
        yperm = y.copy()
        for m in mice:
            idx = np.where(g == m)[0]; yperm[idx] = rng2.permutation(yperm[idx])
        ypn = np.zeros(len(y))
        for tr, te in LeaveOneGroupOut().split(X, yperm, g):
            mm = pipe(); mm.fit(X[tr], yperm[tr]); ypn[te] = mm.predict_proba(X[te])[:, 1]
        null.append(np.mean([roc_auc_score(yperm[g==m], ypn[g==m]) for m in mice if len(set(yperm[g==m]))==2]))
    null = np.array(null); pval = float((1 + np.sum(null >= lomo)) / (1 + len(null)))
    # per-parcel importance (standardized coef, full fit)
    clf = pipe(); clf.fit(X, y)
    coef = np.abs(clf.named_steps["logisticregression"].coef_.ravel())
    top = np.argsort(-coef)[:8].tolist()
    return {"target": target, "n_mice": len(mice), "n_events": int(len(y)//2),
            "within_mouse_auc": float(np.mean(wm)), "within_mouse_sd": float(np.std(wm)),
            "lomo_auc": lomo, "lomo_ci95": ci, "perm_p": pval, "perm_null_mean": float(np.mean(null)),
            "per_mouse_auc": {m: round(v, 3) for m, v in per_mouse.items()},
            "top_parcels": top, "parcel_importance": coef.round(3).tolist()}


def main():
    sess = pick_sessions()
    print(f"mice: {len(sess)}")
    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(build_mouse, sess.items()))
    mouse_data = {tgt: {} for tgt in TARGETS}
    for mouse, per, msg in results:
        print(f"  {mouse}: {msg}")
        if per:
            for tgt, xy in per.items():
                mouse_data[tgt][mouse] = xy
    out = {}
    for tgt in TARGETS:
        if len(mouse_data[tgt]) >= 5:
            print(f"\n=== analyzing {tgt} ({len(mouse_data[tgt])} mice) ===")
            r = analyze(tgt, mouse_data[tgt]); out[tgt] = r
            print(f"  within-mouse {r['within_mouse_auc']:.3f}+/-{r['within_mouse_sd']:.3f} | "
                  f"LOMO {r['lomo_auc']:.3f} CI{r['lomo_ci95']} | perm p={r['perm_p']:.3f}")
    json.dump(out, open("/home/ubuntu/braidyn_results.json", "w"), indent=1)
    print("\nsaved braidyn_results.json")
    print("BRAIDYN_FULL_DONE")


if __name__ == "__main__":
    main()
