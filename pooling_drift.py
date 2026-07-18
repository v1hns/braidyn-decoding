"""Population pooling as drift resistance (DANDI:001425).

Verified core effect (from braidyn_drift.json, per-mouse): a mouse's OWN early-block decoder
ages more across ~2 weeks than a decoder POOLED over the other 24 mice. Pooled asymmetry
+0.024 AUC, p=2e-5, own decoder ages more in 76% of mouse-target pairs, all 4 targets p<0.005.

This script re-derives that with bootstrap CIs AND tests the MECHANISM: is the pooled decoder
robust because it averages out INDIVIDUAL-specific drift (diversity), or just because it has
more training data (volume)? Three analyses off a single streamed feature cache:

  A. AGING ASYMMETRY  -- per mouse: personal aging (WS-WX) vs pooled aging (LS-LX); paired
     sign-flip test + cluster bootstrap over mice. (confirms the headline)
  B. SCALING          -- pooled decoder trained on N in {1,2,4,8,16,24} random other mice;
     aging vs N. Prediction: aging falls monotonically with N.
  C. COUNT-MATCHED    -- the decisive control. At a MATCHED training-event count n, compare a
     decoder trained on events drawn from MANY other mice vs from ONE other mouse. If
     many-mice ages less at equal n, the driver is DIVERSITY, not data volume.

Streams parcellated dF/F only (remfile); caches per-mouse per-block (X,y). Saves pooling_results.json.
"""
import re, json, os, socket
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

socket.setdefaulttimeout(90)
TARGETS = ["state_lever", "lick", "reward", "tone"]
EARLY_MAX, LATE_MIN = 5, 11
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "pooling_features.npz")


# ---------- data access (identical logic to braidyn_drift.py) ----------
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


def _auc(clf, X, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, clf.predict_proba(X)[:, 1]))


def _fit(X, y):
    c = pipe(); c.fit(X, y); return c


def _both(y):
    return len(set(y.tolist())) == 2


def gather():
    if os.path.exists(CACHE):
        print(f"loading cached features {CACHE}", flush=True)
        z = np.load(CACHE, allow_pickle=True)
        return z["early"].item(), z["late"].item()
    task = list_task_sessions()
    jobs = [(m, day, path) for m, sess in task.items() for (day, path) in sess]
    print(f"mice {len(task)}  task-sessions {len(jobs)}", flush=True)
    # target -> mouse -> {day: (X,y)}
    per = {t: defaultdict(dict) for t in TARGETS}
    ok = 0
    with ThreadPoolExecutor(max_workers=24) as ex:
        futs = {ex.submit(build_session, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            mouse, day, p, msg = fut.result()
            if p:
                ok += 1
                for t, xy in p.items():
                    per[t][mouse][day] = xy
            if i % 25 == 0 or i == len(jobs):
                print(f"  streamed {i}/{len(jobs)} usable={ok}", flush=True)
    # pool into early/late blocks per target per mouse
    early = {t: {} for t in TARGETS}; late = {t: {} for t in TARGETS}
    for t in TARGETS:
        for m, sess in per[t].items():
            e = [sess[d] for d in sess if d <= EARLY_MAX]
            l = [sess[d] for d in sess if d >= LATE_MIN]
            if e and l:
                Xe = np.vstack([x for x, _ in e]); ye = np.concatenate([y for _, y in e])
                Xl = np.vstack([x for x, _ in l]); yl = np.concatenate([y for _, y in l])
                if _both(ye) and _both(yl):
                    early[t][m] = (Xe, ye); late[t][m] = (Xl, yl)
    np.savez(CACHE, early=np.array(early, dtype=object), late=np.array(late, dtype=object))
    print(f"cached {CACHE}", flush=True)
    return early, late


# ---------- A. aging asymmetry (bootstrap + paired sign-flip) ----------
def analysis_A(early, late):
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    out = {}
    for t in TARGETS:
        mice = sorted(early[t])
        rows = []
        for m in mice:
            Xe, ye = early[t][m]; Xl, yl = late[t][m]
            others = [o for o in mice if o != m]
            Xtr = np.vstack([early[t][o][0] for o in others])
            ytr = np.concatenate([early[t][o][1] for o in others])
            if not (_both(ye) and _both(yl) and _both(ytr)):
                continue
            # personal: CV on early (WS), train-early test-late (WX)
            ws = float(cross_val_score(pipe(), Xe, ye,
                       cv=StratifiedKFold(5, shuffle=True, random_state=0),
                       scoring="roc_auc").mean())
            pc = _fit(Xe, ye); wx = _auc(pc, Xl, yl)
            # pooled: others-early -> M-early (LS), -> M-late (LX)
            oc = _fit(Xtr, ytr); ls = _auc(oc, Xe, ye); lx = _auc(oc, Xl, yl)
            rows.append((m, ws, wx, ls, lx))
        arr = np.array([(r[1], r[2], r[3], r[4]) for r in rows])
        personal_age = arr[:, 0] - arr[:, 1]      # WS - WX
        pooled_age = arr[:, 2] - arr[:, 3]        # LS - LX
        asym = personal_age - pooled_age
        rng = np.random.default_rng(1)
        null = np.array([(asym * rng.choice([-1, 1], len(asym))).mean() for _ in range(20000)])
        p = float((1 + np.sum(np.abs(null) >= abs(asym.mean()))) / (1 + len(null)))
        bs = [np.mean(rng.choice(asym, len(asym), replace=True)) for _ in range(5000)]
        out[t] = {"n": len(rows),
                  "personal_aging": float(personal_age.mean()),
                  "pooled_aging": float(pooled_age.mean()),
                  "asymmetry": float(asym.mean()),
                  "asym_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                  "p_signflip": p,
                  "frac_personal_ages_more": float(np.mean(asym > 0)),
                  "per_mouse": {r[0]: {"WS": round(r[1], 3), "WX": round(r[2], 3),
                                       "LS": round(r[3], 3), "LX": round(r[4], 3)} for r in rows}}
        print(f"  [A {t}] asym {out[t]['asymmetry']:+.4f} p={p:.4f} "
              f"({out[t]['frac_personal_ages_more']*100:.0f}% own ages more)", flush=True)
    return out


# ---------- B. scaling: pooled aging vs N training mice ----------
def analysis_B(early, late, Ns=(1, 2, 4, 8, 16, 24), reps=8):
    out = {}
    for t in TARGETS:
        mice = sorted(early[t]); res = {N: [] for N in Ns}
        for m in mice:
            Xe, ye = early[t][m]; Xl, yl = late[t][m]
            if not (_both(ye) and _both(yl)):
                continue
            others = [o for o in mice if o != m]
            for N in Ns:
                if N > len(others):
                    continue
                agings = []
                for rep in range(reps):
                    rng = np.random.default_rng(1000 * N + rep)
                    sub = list(rng.choice(others, N, replace=False))
                    Xtr = np.vstack([early[t][o][0] for o in sub])
                    ytr = np.concatenate([early[t][o][1] for o in sub])
                    if not _both(ytr):
                        continue
                    c = _fit(Xtr, ytr)
                    agings.append(_auc(c, Xe, ye) - _auc(c, Xl, yl))  # LS_N - LX_N
                if agings:
                    res[N].append(np.mean(agings))
        out[t] = {int(N): {"pooled_aging": float(np.mean(v)),
                           "sem": float(np.std(v) / max(1, np.sqrt(len(v)))), "n": len(v)}
                  for N, v in res.items() if v}
        curve = " ".join(f"N{N}:{out[t][N]['pooled_aging']:+.3f}" for N in sorted(out[t]))
        print(f"  [B {t}] {curve}", flush=True)
    return out


# ---------- C. count-matched control: diversity vs volume ----------
def analysis_C(early, late, reps=12):
    """At MATCHED training-event count n = size of M's own early block, compare:
       many : n events pooled from MANY other mice
       one  : n events from a SINGLE other mouse
       personal: M's own early block (the n it naturally has).
    Aging = test M-early - test M-late. If many ages less than one at equal n -> diversity."""
    out = {}
    for t in TARGETS:
        mice = sorted(early[t]); rows = {"personal": [], "many": [], "one": []}
        for m in mice:
            Xe, ye = early[t][m]; Xl, yl = late[t][m]
            if not (_both(ye) and _both(yl)):
                continue
            n = len(ye)
            others = [o for o in mice if o != m]
            # personal aging
            pc = _fit(Xe, ye); rows["personal"].append(_auc(pc, Xe, ye) - _auc(pc, Xl, yl))
            # pooled bank from all others
            Xall = np.vstack([early[t][o][0] for o in others])
            yall = np.concatenate([early[t][o][1] for o in others])
            many_a, one_a = [], []
            for rep in range(reps):
                rng = np.random.default_rng(7000 + 13 * rep + hash(m) % 997)
                # many: n samples drawn across the whole pooled bank
                idx = rng.choice(len(yall), min(n, len(yall)), replace=False)
                Xm, ym = Xall[idx], yall[idx]
                if _both(ym):
                    c = _fit(Xm, ym); many_a.append(_auc(c, Xe, ye) - _auc(c, Xl, yl))
                # one: n samples from a single randomly chosen other mouse
                o = str(rng.choice(others))
                Xo, yo = early[t][o]
                if len(yo) >= n and _both(yo):
                    j = rng.choice(len(yo), n, replace=False)
                    Xo2, yo2 = Xo[j], yo[j]
                elif _both(yo):
                    Xo2, yo2 = Xo, yo
                else:
                    continue
                if _both(yo2):
                    c = _fit(Xo2, yo2); one_a.append(_auc(c, Xe, ye) - _auc(c, Xl, yl))
            if many_a:
                rows["many"].append(np.mean(many_a))
            if one_a:
                rows["one"].append(np.mean(one_a))
        summ = {}
        for k, v in rows.items():
            v = np.array(v)
            summ[k] = {"aging": float(v.mean()), "n": len(v),
                       "sem": float(v.std() / max(1, np.sqrt(len(v))))}
        # paired many-vs-one over mice present in both
        common_many = {m: a for m, a in zip([mm for mm in mice], rows["many"])}
        summ["_note"] = "many vs one at matched n; many<one aging => diversity drives robustness"
        out[t] = summ
        print(f"  [C {t}] personal {summ['personal']['aging']:+.3f} | "
              f"many {summ['many']['aging']:+.3f} | one {summ['one']['aging']:+.3f}", flush=True)
    return out


def main():
    early, late = gather()
    for t in TARGETS:
        print(f"{t}: {len(early[t])} mice with both blocks", flush=True)
    out = {"targets": TARGETS,
           "A_aging_asymmetry": analysis_A(early, late),
           "B_scaling": analysis_B(early, late),
           "C_count_matched": analysis_C(early, late)}
    dest = os.path.join(HERE, "pooling_results.json")
    json.dump(out, open(dest, "w"), indent=1)
    print(f"\nsaved {dest}\nPOOLING_DONE", flush=True)


if __name__ == "__main__":
    main()
