"""Independent-cohort check of the split-level bias: Cardin-Higley widefield (figshare 175317).

Different lab, different atlas (23 CCFv3 parcels vs 44 Allen), different task (spontaneous behavior,
not a cued operant task), different behavioral targets (locomotion from a wheel, arousal from pupil).
Each animal is ONE continuous recording rather than a set of sessions, so the analogue of a
"session-held-out" split is a CONTIGUOUS-BLOCK split: neighbouring time windows share slow drift,
arousal and imaging state exactly the way trials within a session do.

Compares, per animal:
  within_shuffle -- 5-fold over randomly shuffled windows           (the biased estimator)
  within_block   -- 5-fold over contiguous time blocks              (the correct one)
  cross          -- train on the other animals, test on this one    (immune by construction)

If the bias reported on BraiDyn is a property of the split rather than of that dataset, the same
one-sided inflation should appear here.

Writes cardin_leakage.json.
"""
import glob, json, os, warnings
import numpy as np
from scipy.io import loadmat
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
WIN = 15          # window length in frames, matching cardin_pilot.py


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
    if img is None:
        arrs = [(k, np.asarray(m[k]).squeeze()) for k in keys]
        arrs = [(k, a) for k, a in arrs if a.ndim == 2]
        img = max(arrs, key=lambda x: x[1].size)[1] if arrs else None
    if img is not None and img.shape[0] > img.shape[1]:
        img = img.T
    return img, find(["pupil"]), find(["wheel", "loco", "speed", "run"])


def windows(img, sig, win=WIN):
    P, T = img.shape
    X, y, t = [], [], []
    med = np.nanmedian(sig)
    for i in range(T // win):
        sl = slice(i * win, (i + 1) * win)
        f = np.nanmean(img[:, sl], 1)
        s = np.nanmean(sig[sl])
        if np.any(np.isnan(f)) or np.isnan(s):
            continue
        X.append(f); y.append(1 if s > med else 0); t.append(i)
    return np.array(X), np.array(y), np.array(t)


def pipe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))


def auc(clf, X, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, clf.predict_proba(X)[:, 1]))


def both(y):
    return len(set(y.tolist())) == 2


def main():
    from sklearn.model_selection import StratifiedKFold
    files = sorted(glob.glob(os.path.expanduser("~/cardin/animal_*.mat")))
    print(f"{len(files)} animal files", flush=True)
    if not files:
        print("NO DATA -- fetch figshare 175317 into ~/cardin/ first"); return

    data = {}
    for f in files:
        img, pupil, wheel = load(f)
        if img is None:
            continue
        for name, sig in [("movement", wheel), ("arousal", pupil)]:
            if sig is None:
                continue
            L = min(img.shape[1], len(sig))
            X, y, t = windows(img[:, :L], sig[:L])
            if len(y) > 40 and both(y):
                data.setdefault(name, {})[os.path.basename(f)] = (X, y, t)

    out = {}
    for target, per in data.items():
        animals = sorted(per)
        rows = []
        for a in animals:
            X, y, t = per[a]
            n = len(y)

            # (a) shuffled 5-fold over windows -- neighbouring windows land in train and test
            sh = []
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
                if both(y[tr]) and both(y[te]):
                    sh.append(auc(pipe().fit(X[tr], y[tr]), X[te], y[te]))

            # (b) contiguous-block 5-fold -- each test fold is one unbroken stretch of time
            bl = []
            edges = np.linspace(0, n, 6).astype(int)
            for i in range(5):
                te = np.zeros(n, bool); te[edges[i]:edges[i + 1]] = True
                if te.sum() < 5 or not (both(y[te]) and both(y[~te])):
                    continue
                bl.append(auc(pipe().fit(X[~te], y[~te]), X[te], y[te]))

            # (c) cross-animal: train on every other animal
            ox = np.vstack([per[o][0] for o in animals if o != a])
            oy = np.concatenate([per[o][1] for o in animals if o != a])
            cr = auc(pipe().fit(ox, oy), X, y) if both(oy) else None

            if sh and bl and cr is not None:
                rows.append({"animal": a, "n_windows": int(n),
                             "within_shuffle": float(np.mean(sh)),
                             "within_block": float(np.mean(bl)),
                             "cross": float(cr)})

        if not rows:
            continue
        ws = np.array([r["within_shuffle"] for r in rows])
        wb = np.array([r["within_block"] for r in rows])
        cx = np.array([r["cross"] for r in rows])
        out[target] = {
            "n_animals": len(rows), "per_animal": rows,
            "within_shuffle": float(ws.mean()), "within_block": float(wb.mean()),
            "cross": float(cx.mean()),
            "inflation": float((ws - wb).mean()),
            "gap_shuffle": float((ws - cx).mean()), "gap_block": float((wb - cx).mean()),
            "frac_inflated": float(np.mean(ws > wb))}
        o = out[target]
        print(f"\n[{target}] n={o['n_animals']} animals", flush=True)
        print(f"  within (shuffled windows) {o['within_shuffle']:.4f}", flush=True)
        print(f"  within (contiguous block) {o['within_block']:.4f}", flush=True)
        print(f"  cross-animal              {o['cross']:.4f}", flush=True)
        print(f"  INFLATION of within arm   {o['inflation']:+.4f}  "
              f"({o['frac_inflated']:.0%} of animals)", flush=True)
        print(f"  within-minus-cross gap    {o['gap_shuffle']:+.4f} (shuffled) vs "
              f"{o['gap_block']:+.4f} (block)", flush=True)

    dest = os.path.join(HERE, "cardin_leakage.json")
    json.dump(out, open(dest, "w"), indent=1)
    print(f"\nsaved {dest}\nCARDIN_LEAKAGE_DONE", flush=True)


if __name__ == "__main__":
    main()
