"""BraiDyn-BC leave-mouse-out cortex-wide decoding pilot (DANDI:001425).

Streams ONLY the parcellated dF/F (18000x44 Allen ROIs) + behavioral signals from each
session's NWB on S3 (remfile) — never downloads the 5GB widefield movie. Decodes lever-pull
onset (evoked cortical response) vs baseline.

Positive control: within-mouse pull-vs-baseline (must be high, ~0.9 AUC).
Novel axis: LEAVE-ONE-MOUSE-OUT (44 shared atlas parcels -> naturally aligned across mice).
Plus permutation baseline. Reports the generalization gap.
"""
import re, sys
import numpy as np


def sessions_with_events():
    """Inventory: return {mouse: [asset_paths]} for behavior-bearing sessions (prefer task)."""
    from dandi.dandiapi import DandiAPIClient
    with DandiAPIClient() as c:
        d = c.get_dandiset("001425", "draft")
        assets = [a for a in d.get_assets() if a.path.endswith(".nwb")]
    by_mouse = {}
    for a in assets:
        m = re.search(r"(sub-[^/]+?)/", a.path)
        mouse = m.group(1) if m else "?"
        by_mouse.setdefault(mouse, []).append(a.path)
    # prefer non-resting-state (task) sessions; fall back to any behavior session
    picks = {}
    for mouse, paths in by_mouse.items():
        task = [p for p in paths if "resting-state" not in p and "behavior" in p]
        rest = [p for p in paths if "behavior" in p]
        chosen = (task or rest)
        if chosen:
            picks[mouse] = sorted(chosen)
    return picks


def stream_nwb(path):
    import remfile, h5py, pynwb
    from dandi.dandiapi import DandiAPIClient
    with DandiAPIClient() as c:
        d = c.get_dandiset("001425", "draft")
        asset = d.get_asset_by_path(path)
        url = asset.get_content_url(follow_redirects=1, strip_query=True)
    f = h5py.File(remfile.File(url), "r")
    io = pynwb.NWBHDF5IO(file=f, load_namespaces=True)
    nwb = io.read()
    dff = np.asarray(nwb.processing["ophys"]["DfOverF"]["dFF"].data)  # (T,44)
    dn = nwb.processing["downsampled"]
    def sig(name):
        try:
            return np.asarray(dn[name].data).ravel()
        except Exception:
            return None
    beh = {k: sig(k) for k in ("state_lever", "lever", "reward", "tone", "lick")}
    ts = nwb.processing["ophys"]["DfOverF"]["dFF"].timestamps
    rate = 1.0 / np.median(np.diff(np.asarray(ts[:200]))) if ts is not None else 30.0
    return dff, beh, rate


def onsets(sig, rate):
    """rising edges of a (thresholded) signal -> onset indices, min 1s apart."""
    if sig is None or np.all(sig == sig[0]):
        return np.array([], int)
    x = sig.astype(float)
    thr = (np.nanmax(x) + np.nanmin(x)) / 2 if np.nanmax(x) > np.nanmin(x) else 0.5
    b = (x > thr).astype(int)
    on = np.where(np.diff(b) == 1)[0] + 1
    if len(on) == 0:
        return on
    keep = [on[0]]
    for o in on[1:]:
        if o - keep[-1] > rate:
            keep.append(o)
    return np.array(keep, int)


def windows(dff, ev, rate, pre=0.5, post=1.0):
    """evoked feature = mean post-onset dFF minus pre-onset baseline, per parcel (44-d)."""
    T = dff.shape[0]; a = int(pre * rate); b = int(post * rate)
    feats = []
    for e in ev:
        if e - a < 0 or e + b >= T:
            continue
        base = dff[e - a:e].mean(0); resp = dff[e:e + b].mean(0)
        feats.append(resp - base)
    return np.array(feats)


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_score
    from sklearn.metrics import roc_auc_score
    picks = sessions_with_events()
    print(f"mice with behavior sessions: {len(picks)}")
    X, y, mouse = [], [], []
    used = 0
    for mi, (m, paths) in enumerate(sorted(picks.items())):
        if used >= 10:
            break
        p = paths[0]
        try:
            dff, beh, rate = stream_nwb(p)
        except Exception as e:
            print("  stream err", m, str(e)[:70]); continue
        ev = onsets(beh.get("state_lever"), rate)
        src = "state_lever"
        for alt in ("lever", "reward", "tone"):
            if len(ev) < 8:
                ev = onsets(beh.get(alt), rate); src = alt
        if len(ev) < 8:
            print(f"  {m}: too few events ({len(ev)}) — skip"); continue
        pos = windows(dff, ev, rate)
        # baseline windows: random points >2s from any event
        rng = np.random.default_rng(0); T = dff.shape[0]
        far = [t for t in rng.integers(int(2*rate), T-int(2*rate), size=len(ev)*2)
               if np.min(np.abs(ev - t)) > 2*rate][:len(pos)]
        neg = windows(dff, np.array(far), rate)
        n = min(len(pos), len(neg))
        if n < 6:
            print(f"  {m}: too few windows — skip"); continue
        Xi = np.vstack([pos[:n], neg[:n]]); yi = np.r_[np.ones(n), np.zeros(n)]
        X.append(Xi); y.append(yi); mouse += [m]*len(yi); used += 1
        print(f"  {m}: rate~{rate:.0f}Hz, {len(ev)} {src} events, {n} pos/{n} neg windows")
    if used < 3:
        print("INSUFFICIENT mice for leave-mouse-out"); print("BRAIDYN_PILOT_DONE"); return
    X = np.vstack(X); y = np.concatenate(y); mouse = np.array(mouse)
    print(f"\ntotal: {X.shape} | mice used: {len(set(mouse))}")

    pipe = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    # POSITIVE CONTROL: within-mouse pooled 5-fold (per mouse), averaged
    wm = []
    for m in sorted(set(mouse)):
        idx = mouse == m
        if len(set(y[idx])) < 2 or idx.sum() < 12:
            continue
        try:
            sc = cross_val_score(pipe(), X[idx], y[idx], cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc")
            wm.append(sc.mean())
        except Exception:
            pass
    print(f"\n== POSITIVE CONTROL within-mouse pull-vs-baseline AUC: {np.mean(wm):.3f} +/- {np.std(wm):.3f} (n={len(wm)} mice) ==")

    # NOVEL AXIS: leave-one-mouse-out
    from sklearn.base import clone
    yp = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, mouse):
        m = pipe(); m.fit(X[tr], y[tr]); yp[te] = m.predict_proba(X[te])[:, 1]
    auc_lomo = roc_auc_score(y, yp)
    per = {m: roc_auc_score(y[mouse==m], yp[mouse==m]) for m in sorted(set(mouse)) if len(set(y[mouse==m]))==2}
    print(f"== NOVEL AXIS leave-one-mouse-out AUC: {auc_lomo:.3f} ==")
    print("   per-held-out-mouse AUC:", {k: round(v,2) for k,v in per.items()})
    # permutation baseline
    rng = np.random.default_rng(1); null = []
    for _ in range(200):
        yperm = rng.permutation(y); ypn = np.zeros(len(y))
        for tr, te in LeaveOneGroupOut().split(X, yperm, mouse):
            mm = pipe(); mm.fit(X[tr], yperm[tr]); ypn[te] = mm.predict_proba(X[te])[:,1]
        null.append(roc_auc_score(yperm, ypn))
    null = np.array(null); pval = (1+np.sum(null >= auc_lomo))/(1+len(null))
    print(f"   permutation null AUC {np.mean(null):.3f}, p={pval:.3f}")
    print("BRAIDYN_PILOT_DONE")


if __name__ == "__main__":
    main()
