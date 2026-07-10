"""Cardin-Higley widefield cross-animal replication pilot (Benisty/Higley 2023, figshare 175317).

6 mice, each: CCFv3 parcel activity (~23 parcels x T) + pupil/face/wheel behavior. Tests whether
BraiDyn's finding (cortex-wide behavioral state decodes ACROSS animals in shared atlas space)
replicates on an INDEPENDENT widefield cohort. Decode movement (wheel) and arousal (pupil):
  - within-mouse: block-wise CV (contiguous time blocks -> no autocorrelation leakage)
  - leave-one-mouse-out: 6-fold (the cross-animal axis)
"""
import glob, os, warnings
import numpy as np
from scipy.io import loadmat
warnings.filterwarnings("ignore")


def load(path):
    m = loadmat(path)
    keys = [k for k in m if not k.startswith("__")]
    def find(cands, need2d=False):
        for k in keys:
            if any(c in k.lower() for c in cands):
                a = np.asarray(m[k]).squeeze()
                if need2d and a.ndim == 2:
                    return a
                if not need2d and a.ndim <= 1:
                    return a.ravel()
        return None
    img = find(["imaging", "parcel", "signal", "activity", "dff", "ccf"], need2d=True)
    if img is None:  # fallback: largest 2D array
        arrs = [(k, np.asarray(m[k]).squeeze()) for k in keys]
        arrs = [(k, a) for k, a in arrs if a.ndim == 2]
        img = max(arrs, key=lambda x: x[1].size)[1] if arrs else None
    if img is not None and img.shape[0] > img.shape[1]:
        img = img.T   # want (parcels, time)
    pupil = find(["pupil"]); wheel = find(["wheel", "loco", "speed", "run"]); face = find(["face", "whisk", "motion"])
    return img, pupil, wheel, face, keys


def windows(img, sig, hi_is_state=True, win=15):
    """window-average parcel features; binary label = signal above/below its median in that window."""
    P, T = img.shape
    n = T // win
    X, y = [], []
    med = np.nanmedian(sig)
    for i in range(n):
        sl = slice(i*win, (i+1)*win)
        f = np.nanmean(img[:, sl], 1)
        s = np.nanmean(sig[sl])
        if np.any(np.isnan(f)) or np.isnan(s):
            continue
        X.append(f); y.append(1 if s > med else 0)
    return np.array(X), np.array(y)


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
    from sklearn.metrics import roc_auc_score
    files = sorted(glob.glob(os.path.expanduser("~/cardin/animal_*.mat")))
    print(f"{len(files)} animal files")
    # first-file key probe
    if files:
        _, _, _, _, keys = load(files[0]); print("mat keys:", keys)
    for target_name, sel in [("movement(wheel)", "wheel"), ("arousal(pupil)", "pupil")]:
        Xall, yall, g = [], [], []
        for ai, f in enumerate(files):
            img, pupil, wheel, face, _ = load(f)
            sig = wheel if sel == "wheel" else pupil
            if img is None or sig is None or len(sig) != img.shape[1]:
                # try to align length
                if img is not None and sig is not None:
                    L = min(img.shape[1], len(sig)); img = img[:, :L]; sig = sig[:L]
                else:
                    print(f"  animal {ai+1}: missing {sel}"); continue
            X, y = windows(img, sig)
            if len(set(y)) < 2 or len(y) < 20:
                continue
            # balance
            Xall.append(X); yall.append(y); g += [ai]*len(y)
        if len(Xall) < 4:
            print(f"{target_name}: too few animals"); continue
        X = np.vstack(Xall); y = np.concatenate(yall); g = np.array(g)
        print(f"\n=== {target_name}: {X.shape}, {len(set(g))} mice, {X.shape[1]} parcels ===")
        pipe = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
        # within-mouse block CV
        wm = []
        for m in sorted(set(g)):
            idx = np.where(g == m)[0]
            blocks = (np.arange(len(idx)) // max(1, len(idx)//5))  # contiguous time blocks
            yp = np.zeros(len(idx))
            ok = True
            for b in sorted(set(blocks)):
                tr = blocks != b; te = blocks == b
                if len(set(y[idx][tr])) < 2: ok = False; break
                mm = pipe(); mm.fit(X[idx][tr], y[idx][tr]); yp[te] = mm.predict_proba(X[idx][te])[:,1]
            if ok and len(set(y[idx]))==2:
                wm.append(roc_auc_score(y[idx], yp))
        # LOMO
        yp = np.zeros(len(y))
        for tr, te in LeaveOneGroupOut().split(X, y, g):
            mm = pipe(); mm.fit(X[tr], y[tr]); yp[te] = mm.predict_proba(X[te])[:,1]
        per = {int(m): round(roc_auc_score(y[g==m], yp[g==m]),3) for m in sorted(set(g)) if len(set(y[g==m]))==2}
        lomo = float(np.mean(list(per.values())))
        print(f"  within-mouse (block-CV) AUC: {np.mean(wm):.3f} +/- {np.std(wm):.3f} (n={len(wm)})")
        print(f"  LEAVE-ONE-MOUSE-OUT AUC: {lomo:.3f} | per-mouse {per}")
    print("CARDIN_DONE")


if __name__ == "__main__":
    main()
