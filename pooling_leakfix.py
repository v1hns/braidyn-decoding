"""Leakage-corrected re-run of the pooling/drift analyses (DANDI:001425).

Two bugs were found in pooling_drift.py, both of which inflate the PERSONAL arm only --
i.e. they inflate an asymmetry measurement from one side:

  A) analysis_A estimated the personal decoder's early-block accuracy (WS) with
     StratifiedKFold(shuffle=True) over trials POOLED ACROSS SESSIONS. Trials from the
     same recording session share imaging conditions, hemodynamic state, arousal and are
     temporally autocorrelated, so the decoder trains on session s and is tested on other
     trials of session s. WS is inflated -> personal aging (WS-WX) inflated -> asymmetry
     inflated. The pooled arm trains on OTHER MICE and cannot leak.
     Secondary, opposing bias: WS trained on 80% of early while WX trained on 100%.

  C) analysis_C (the count-matched control) evaluated the self arm on its OWN training
     data: `pc = _fit(Xe, ye); _auc(pc, Xe, ye) - _auc(pc, Xl, yl)`. The first term is
     TRAINING accuracy, with no cross-validation at all, while the one/many arms are
     genuinely held out on that same data. This is the paper's decisive experiment.

  analysis_B (scaling vs N training mice) never lets the decoder see the held-out mouse
  and is clean; it is re-run here unchanged as a control.

This script runs the ORIGINAL (leaky) and CORRECTED protocols on IDENTICAL cached features,
so the difference is a direct measurement of the leak rather than an inference.

CORRECTED designs
  A_fixed: for each held-out early session s of mouse m,
             clf = fit(early \\ {s});  ws_s = AUC(clf, s);  wx_s = AUC(clf, late)
           WS = mean_s ws_s, WX = mean_s wx_s  -> identical training sets, kills both biases.
           clf_pool = fit(other mice's early); LS = mean_s AUC(clf_pool, s); LX = AUC(clf_pool, late)
  C_fixed: for each held-out early session s, with n = |early \\ {s}| events,
           all three arms train on n events and are scored on the SAME held-out s and late:
             self = n events from early\\{s} | one = n from a single other mouse | many = n across others

Writes pooling_leakfix.json. Feature cache: leakfix_features.npz (per target/mouse/day).
"""
import re, json, os, socket
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

socket.setdefaulttimeout(90)
TARGETS = ["state_lever", "lick", "reward", "tone"]
EARLY_MAX, LATE_MIN = 5, 11
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "leakfix_features.npz")


# ---------------- streaming (identical to full_decay.py / pooling_drift.py) ----------------
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
    return mouse, day, per, f"{list(per.keys())}"


def gather():
    """target -> mouse -> {day: (X, y)}  -- session identity PRESERVED (the original vstacked it away)."""
    if os.path.exists(CACHE):
        print(f"loading cached features {CACHE}", flush=True)
        return np.load(CACHE, allow_pickle=True)["per"].item()
    task = list_task_sessions()
    jobs = [(m, day, path) for m, sess in task.items() for (day, path) in sess]
    print(f"mice {len(task)}  task-sessions {len(jobs)}", flush=True)
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
    per = {t: dict(v) for t, v in per.items()}
    np.savez(CACHE, per=np.array(per, dtype=object))
    print(f"cached {CACHE}", flush=True)
    return per


# ---------------- helpers ----------------
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


def _pool(sess, days):
    xs = [sess[d][0] for d in days if d in sess]; ys = [sess[d][1] for d in days if d in sess]
    if not xs:
        return None
    return np.vstack(xs), np.concatenate(ys)


def _blocks(sess):
    e_days = sorted(d for d in sess if d <= EARLY_MAX)
    l = _pool(sess, [d for d in sess if d >= LATE_MIN])
    e = _pool(sess, e_days)
    return e_days, e, l


def _sign_test(vals, seed=1, n=20000):
    v = np.array(vals, float); rng = np.random.default_rng(seed)
    null = np.array([(v * rng.choice([-1, 1], len(v))).mean() for _ in range(n)])
    p = float((1 + np.sum(np.abs(null) >= abs(v.mean()))) / (1 + len(null)))
    bs = [np.mean(rng.choice(v, len(v), replace=True)) for _ in range(5000)]
    return {"mean": float(v.mean()), "p_signflip": p, "n": len(v),
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "frac_positive": float(np.mean(v > 0))}


# ---------------- A: aging asymmetry, leaky vs fixed ----------------
def analysis_A(per):
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    out = {}
    for t in TARGETS:
        pm = per[t]
        usable = {}
        for m, sess in pm.items():
            e_days, e, l = _blocks(sess)
            if e is None or l is None or len(e_days) < 2:
                continue
            if _both(e[1]) and _both(l[1]):
                usable[m] = (e_days, e, l)
        mice = sorted(usable)
        leaky, fixed = [], []
        for m in mice:
            e_days, (Xe, ye), (Xl, yl) = usable[m]
            sess = pm[m]
            others = [o for o in mice if o != m]
            Xtr = np.vstack([usable[o][1][0] for o in others])
            ytr = np.concatenate([usable[o][1][1] for o in others])
            if not _both(ytr):
                continue
            clf_pool = _fit(Xtr, ytr)

            # ---- ORIGINAL (leaky): trial-shuffled 5-fold over pooled early sessions ----
            ws_leak = float(cross_val_score(pipe(), Xe, ye,
                            cv=StratifiedKFold(5, shuffle=True, random_state=0),
                            scoring="roc_auc").mean())
            wx_leak = _auc(_fit(Xe, ye), Xl, yl)
            ls_leak = _auc(clf_pool, Xe, ye); lx_leak = _auc(clf_pool, Xl, yl)
            leaky.append((m, (ws_leak - wx_leak) - (ls_leak - lx_leak),
                          ws_leak, wx_leak, ls_leak, lx_leak))

            # ---- CORRECTED: leave-one-early-SESSION-out, matched training sets ----
            ws_s, wx_s, ls_s = [], [], []
            for s in e_days:
                tr = _pool(sess, [d for d in e_days if d != s])
                if tr is None or not _both(tr[1]):
                    continue
                Xs, ys = sess[s]
                if not _both(ys):
                    continue
                clf_p = _fit(tr[0], tr[1])
                ws_s.append(_auc(clf_p, Xs, ys))
                wx_s.append(_auc(clf_p, Xl, yl))
                ls_s.append(_auc(clf_pool, Xs, ys))     # pooled scored on the SAME held-out session
            if not ws_s:
                continue
            WS, WX, LS = float(np.mean(ws_s)), float(np.mean(wx_s)), float(np.mean(ls_s))
            LX = _auc(clf_pool, Xl, yl)
            fixed.append((m, (WS - WX) - (LS - LX), WS, WX, LS, LX))

        def pack(rows):
            a = _sign_test([r[1] for r in rows])
            a.update({"personal_aging": float(np.mean([r[2] - r[3] for r in rows])),
                      "pooled_aging": float(np.mean([r[4] - r[5] for r in rows])),
                      "WS": float(np.mean([r[2] for r in rows])),
                      "WX": float(np.mean([r[3] for r in rows])),
                      "LS": float(np.mean([r[4] for r in rows])),
                      "LX": float(np.mean([r[5] for r in rows])),
                      "per_mouse": {r[0]: {"WS": round(r[2], 4), "WX": round(r[3], 4),
                                           "LS": round(r[4], 4), "LX": round(r[5], 4)} for r in rows}})
            return a
        out[t] = {"leaky": pack(leaky), "fixed": pack(fixed)}
        L, F = out[t]["leaky"], out[t]["fixed"]
        print(f"  [A {t:11s}] LEAKY asym {L['mean']:+.4f} (p={L['p_signflip']:.4f}, WS {L['WS']:.4f})  "
              f"->  FIXED asym {F['mean']:+.4f} (p={F['p_signflip']:.4f}, WS {F['WS']:.4f})", flush=True)
    return out


# ---------------- B: scaling (already clean) ----------------
def analysis_B(per, Ns=(1, 2, 4, 8, 16, 24), reps=8):
    out = {}
    for t in TARGETS:
        pm = per[t]
        usable = {}
        for m, sess in pm.items():
            e_days, e, l = _blocks(sess)
            if e is not None and l is not None and _both(e[1]) and _both(l[1]):
                usable[m] = (e, l)
        mice = sorted(usable); res = {N: [] for N in Ns}
        for m in mice:
            (Xe, ye), (Xl, yl) = usable[m]
            others = [o for o in mice if o != m]
            for N in Ns:
                if N > len(others):
                    continue
                ag = []
                for rep in range(reps):
                    rng = np.random.default_rng(1000 * N + rep)
                    sub = list(rng.choice(others, N, replace=False))
                    Xtr = np.vstack([usable[o][0][0] for o in sub])
                    ytr = np.concatenate([usable[o][0][1] for o in sub])
                    if not _both(ytr):
                        continue
                    c = _fit(Xtr, ytr)
                    ag.append(_auc(c, Xe, ye) - _auc(c, Xl, yl))
                if ag:
                    res[N].append(float(np.mean(ag)))
        out[t] = {int(N): {"pooled_aging": float(np.mean(v)),
                           "sem": float(np.std(v) / max(1, np.sqrt(len(v)))), "n": len(v)}
                  for N, v in res.items() if v}
        print(f"  [B {t:11s}] " + " ".join(f"N{N}:{out[t][N]['pooled_aging']:+.3f}"
                                           for N in sorted(out[t])), flush=True)
    return out


# ---------------- C: count-matched control, leaky vs fixed ----------------
def analysis_C(per, reps=12):
    out = {}
    for t in TARGETS:
        pm = per[t]
        usable = {}
        for m, sess in pm.items():
            e_days, e, l = _blocks(sess)
            if e is None or l is None or len(e_days) < 2:
                continue
            if _both(e[1]) and _both(l[1]):
                usable[m] = (e_days, e, l)
        mice = sorted(usable)
        leaky = {"self": [], "one": [], "many": []}
        fixed = {"self": [], "one": [], "many": []}
        for m in mice:
            e_days, (Xe, ye), (Xl, yl) = usable[m]
            sess = pm[m]
            others = [o for o in mice if o != m]
            Xall = np.vstack([usable[o][1][0] for o in others])
            yall = np.concatenate([usable[o][1][1] for o in others])

            # ---- ORIGINAL (leaky): self scored on its OWN training data ----
            pc = _fit(Xe, ye)
            leaky["self"].append(_auc(pc, Xe, ye) - _auc(pc, Xl, yl))
            n = len(ye)
            ma, oa = [], []
            for rep in range(reps):
                rng = np.random.default_rng(7000 + 13 * rep + hash(m) % 997)
                idx = rng.choice(len(yall), min(n, len(yall)), replace=False)
                if _both(yall[idx]):
                    c = _fit(Xall[idx], yall[idx])
                    ma.append(_auc(c, Xe, ye) - _auc(c, Xl, yl))
                o = str(rng.choice(others)); Xo, yo = usable[o][1]
                if len(yo) >= n and _both(yo):
                    j = rng.choice(len(yo), n, replace=False); Xo2, yo2 = Xo[j], yo[j]
                else:
                    Xo2, yo2 = Xo, yo
                if _both(yo2):
                    c = _fit(Xo2, yo2)
                    oa.append(_auc(c, Xe, ye) - _auc(c, Xl, yl))
            if ma: leaky["many"].append(float(np.mean(ma)))
            if oa: leaky["one"].append(float(np.mean(oa)))

            # ---- CORRECTED: every arm trains on n events, scored on the SAME held-out session ----
            f_self, f_one, f_many = [], [], []
            for s in e_days:
                tr = _pool(sess, [d for d in e_days if d != s])
                if tr is None or not _both(tr[1]):
                    continue
                Xs, ys = sess[s]
                if not _both(ys):
                    continue
                n_s = len(tr[1])                      # the self arm's natural budget
                c = _fit(tr[0], tr[1])
                f_self.append(_auc(c, Xs, ys) - _auc(c, Xl, yl))
                mm, oo = [], []
                for rep in range(reps):
                    rng = np.random.default_rng(9000 + 13 * rep + 7 * s + hash(m) % 997)
                    k = min(n_s, len(yall))
                    idx = rng.choice(len(yall), k, replace=False)
                    if _both(yall[idx]):
                        cm = _fit(Xall[idx], yall[idx])
                        mm.append(_auc(cm, Xs, ys) - _auc(cm, Xl, yl))
                    o = str(rng.choice(others)); Xo, yo = usable[o][1]
                    if len(yo) >= n_s and _both(yo):
                        j = rng.choice(len(yo), n_s, replace=False); Xo2, yo2 = Xo[j], yo[j]
                    else:
                        Xo2, yo2 = Xo, yo
                    if _both(yo2):
                        co = _fit(Xo2, yo2)
                        oo.append(_auc(co, Xs, ys) - _auc(co, Xl, yl))
                if mm: f_many.append(float(np.mean(mm)))
                if oo: f_one.append(float(np.mean(oo)))
            if f_self: fixed["self"].append(float(np.mean(f_self)))
            if f_many: fixed["many"].append(float(np.mean(f_many)))
            if f_one:  fixed["one"].append(float(np.mean(f_one)))

        def pack(d):
            return {k: {"aging": float(np.mean(v)), "n": len(v),
                        "sem": float(np.std(v) / max(1, np.sqrt(len(v))))} for k, v in d.items() if v}
        out[t] = {"leaky": pack(leaky), "fixed": pack(fixed)}
        L, F = out[t]["leaky"], out[t]["fixed"]
        print(f"  [C {t:11s}] LEAKY self {L['self']['aging']:+.4f} one {L['one']['aging']:+.4f} "
              f"many {L['many']['aging']:+.4f}  ->  FIXED self {F['self']['aging']:+.4f} "
              f"one {F['one']['aging']:+.4f} many {F['many']['aging']:+.4f}", flush=True)
    return out


def main():
    per = gather()
    print("\n=== A: aging asymmetry (leaky vs fixed) ===", flush=True)
    A = analysis_A(per)
    print("\n=== B: scaling with N training mice (control, already clean) ===", flush=True)
    B = analysis_B(per)
    print("\n=== C: count-matched control (leaky vs fixed) ===", flush=True)
    C = analysis_C(per)

    # pooled-over-all-targets asymmetry, both protocols
    pooled = {}
    for k in ("leaky", "fixed"):
        vals = []
        for t in TARGETS:
            pmv = A[t][k]["per_mouse"]
            vals += [(v["WS"] - v["WX"]) - (v["LS"] - v["LX"]) for v in pmv.values()]
        pooled[k] = _sign_test(vals, seed=3)
        print(f"POOLED[{k:5s}] asym {pooled[k]['mean']:+.4f}  p={pooled[k]['p_signflip']:.6f}  "
              f"frac>0={pooled[k]['frac_positive']:.2%}  n={pooled[k]['n']}", flush=True)

    dest = os.path.join(HERE, "pooling_leakfix.json")
    json.dump({"A": A, "B": B, "C": C, "pooled_asymmetry": pooled}, open(dest, "w"), indent=1)
    print(f"\nsaved {dest}\nLEAKFIX_DONE", flush=True)


if __name__ == "__main__":
    main()
