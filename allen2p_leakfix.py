"""Leakage-corrected protocol on the Allen Visual Behavior 2-photon replication cohort.

`allen2p_pilot.py` established the aging asymmetry on this cohort (+0.0229, p=0.0031, 74% of
23 mice) using leave-one-early-session-out for the personal arm -- i.e. the corrected design for
analysis A. It was never run through the REST of the protocol that the BraiDyn result had to pass
(see HANDOFF.md 2a). This script closes that gap. Three things, mirroring pooling_leakfix.py so the
two datasets are compared on identical designs:

  A  aging asymmetry, LEAKY vs FIXED side by side on identical cached features.
     leaky = the original pooling_drift.py bug: early sessions vstacked (session identity
             destroyed), WS estimated by StratifiedKFold(shuffle=True) over pooled trials, so the
             personal decoder trains on session s and is scored on other trials of session s.
             The pooled arm trains on other mice and cannot leak -> inflates ONE side of the
             asymmetry. Secondary opposing bias: WS trained on 80% of early, WX on 100%.
     fixed = leave-one-early-SESSION-out with matched training sets (the pilot's design).
     The gap between them measures the inflation this dataset would have had.

  C  count-matched control. The decisive experiment: if pooling only wins because it has more
     training data, then at EQUAL training-event count the advantage should vanish. For each
     held-out early session s, n = |early \\ {s}| events, and all three arms train on n events and
     are scored on the SAME held-out s and the same late block:
         self = n events from early \\ {s}   one = n from ONE other mouse   many = n across others
     The leaky variant (self scored on its own training data, uncross-validated) is run alongside,
     since that was bug C in the BraiDyn code.

  D  inclusion-rule sensitivity. The pilot required >=4 usable sessions and kept 23 of 25 mice.
     All 25 containers have >=5 sessions on disk, so the 2 losses are per-experiment QC failures
     (cell count / trial count), not short containers. Sweeping the threshold shows whether the
     result depends on where that line is drawn.

Features are identical to the pilot (27-dim permutation-invariant summary of the per-neuron evoked
response) and are cached to allen2p_features.npz on the first run, so A/C/D are re-runnable without
re-streaming. Streams NWB from S3; never downloads a whole file. Writes allen2p_leakfix.json.
"""
import json, os, socket, sys, zlib
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.argv = [sys.argv[0], sys.argv[1] if len(sys.argv) > 1 else "25"]
import allen2p_pilot as pilot          # cohort(), build(), pipe(), auc() -- same features, one source

socket.setdefaulttimeout(120)
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "allen2p_features.npz")
MIN_SESSIONS = 4                        # the pilot's inclusion rule; swept in analysis D
REPS = 12                               # resamples per held-out session in the count-matched control


# ---------------- features (streamed once, then cached) ----------------
def gather():
    """mouse -> {day: (X, y, n_neurons)}, session identity PRESERVED."""
    if os.path.exists(CACHE):
        print(f"loading cached features {CACHE}", flush=True)
        return np.load(CACHE, allow_pickle=True)["per"].item()
    sel = pilot.cohort()
    jobs = [(s["mouse"], d, e) for s in sel for d, e in s["sessions"]]
    print(f"cohort {len(sel)} mice / {len(jobs)} experiments -- streaming", flush=True)
    per, drops = defaultdict(dict), []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(pilot.build, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            m, day, eid, res, msg = fut.result()
            if res:
                per[m][day] = res
            else:
                drops.append({"mouse": m, "eid": eid, "reason": msg})
            if i % 20 == 0 or i == len(jobs):
                print(f"  streamed {i}/{len(jobs)} usable={sum(len(v) for v in per.values())}", flush=True)
    per = {"per": dict(per), "drops": drops}
    np.savez(CACHE, per=np.array(per, dtype=object))
    print(f"cached {CACHE}  ({len(drops)} experiments dropped)", flush=True)
    return per


# ---------------- helpers (same semantics as pooling_leakfix.py) ----------------
def _fit(X, y):
    c = pilot.pipe(); c.fit(X, y); return c


def _auc(clf, X, y):
    return pilot.auc(clf, X, y)


def _both(y):
    return len(set(np.asarray(y).tolist())) == 2


def _sign_test(vals, seed=1, n=20000):
    v = np.array(vals, float); rng = np.random.default_rng(seed)
    null = np.array([(v * rng.choice([-1, 1], len(v))).mean() for _ in range(n)])
    p = float((1 + np.sum(np.abs(null) >= abs(v.mean()))) / (1 + len(null)))
    bs = [np.mean(rng.choice(v, len(v), replace=True)) for _ in range(5000)]
    return {"mean": float(v.mean()), "p_signflip": p, "n": len(v),
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "frac_positive": float(np.mean(v > 0))}


def blocks(data, min_sessions=MIN_SESSIONS):
    """mouse -> (early_days, [per-session (X,y)], (X_late, y_late)); early/late split by date."""
    out = {}
    for m in data:
        days = sorted(data[m])
        if len(days) < min_sessions:
            continue
        h = max(2, len(days) // 2)
        e = [(data[m][d][0], data[m][d][1]) for d in days[:h]]
        lx = [data[m][d][0] for d in days[h:]]; ly = [data[m][d][1] for d in days[h:]]
        if not lx or len(e) < 2:
            continue
        Xl, yl = np.vstack(lx), np.concatenate(ly)
        if not _both(yl) or not all(_both(y) for _, y in e):
            continue
        out[m] = (days[:h], e, (Xl, yl))
    return out


def _seed(m):
    """Python's str hash is salted per process; crc32 keeps the resamples reproducible."""
    return zlib.crc32(str(m).encode()) % 997


def others_early(bl, m):
    o = [x for x in bl if x != m]
    X = np.vstack([x for k in o for x, _ in bl[k][1]])
    y = np.concatenate([yy for k in o for _, yy in bl[k][1]])
    return o, X, y


# ---------------- A: aging asymmetry, leaky vs fixed ----------------
def analysis_A(bl, verbose=True):
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    mice = sorted(bl)
    leaky, fixed = [], []
    for m in mice:
        edays, elist, (Xl, yl) = bl[m]
        _, Xo, yo = others_early(bl, m)
        if not _both(yo):
            continue
        clf_pool = _fit(Xo, yo)
        Xe = np.vstack([x for x, _ in elist]); ye = np.concatenate([y for _, y in elist])

        # ---- LEAKY: early sessions vstacked, trial-shuffled 5-fold ----
        ws_l = float(cross_val_score(pilot.pipe(), Xe, ye,
                                     cv=StratifiedKFold(5, shuffle=True, random_state=0),
                                     scoring="roc_auc").mean())
        wx_l = _auc(_fit(Xe, ye), Xl, yl)
        ls_l, lx_l = _auc(clf_pool, Xe, ye), _auc(clf_pool, Xl, yl)
        leaky.append((m, ws_l, wx_l, ls_l, lx_l))

        # ---- FIXED: leave-one-early-SESSION-out, matched training sets ----
        ws, wx, ls = [], [], []
        for i in range(len(elist)):
            tr = [elist[j] for j in range(len(elist)) if j != i]
            Xt = np.vstack([x for x, _ in tr]); yt = np.concatenate([y for _, y in tr])
            Xh, yh = elist[i]
            if not _both(yt) or not _both(yh):
                continue
            c = _fit(Xt, yt)
            ws.append(_auc(c, Xh, yh)); wx.append(_auc(c, Xl, yl))
            ls.append(_auc(clf_pool, Xh, yh))      # pooled scored on the SAME held-out session
        if not ws:
            continue
        fixed.append((m, float(np.mean(ws)), float(np.mean(wx)), float(np.mean(ls)),
                      _auc(clf_pool, Xl, yl)))

    def pack(rows):
        d = _sign_test([(r[1] - r[2]) - (r[3] - r[4]) for r in rows], seed=3)
        d.update({"WS": float(np.mean([r[1] for r in rows])), "WX": float(np.mean([r[2] for r in rows])),
                  "LS": float(np.mean([r[3] for r in rows])), "LX": float(np.mean([r[4] for r in rows])),
                  "personal_aging": float(np.mean([r[1] - r[2] for r in rows])),
                  "pooled_aging": float(np.mean([r[3] - r[4] for r in rows])),
                  "per_mouse": {r[0]: {"WS": round(r[1], 4), "WX": round(r[2], 4),
                                       "LS": round(r[3], 4), "LX": round(r[4], 4),
                                       "asymmetry": round((r[1] - r[2]) - (r[3] - r[4]), 4)}
                                for r in rows}})
        return d

    out = {"leaky": pack(leaky), "fixed": pack(fixed)}
    if verbose:
        for k in ("leaky", "fixed"):
            o = out[k]
            print(f"  [A {k:5s}] asym {o['mean']:+.4f} [{o['ci95'][0]:+.4f},{o['ci95'][1]:+.4f}] "
                  f"p={o['p_signflip']:.4f}  {o['frac_positive']:.0%} of {o['n']} mice  |  "
                  f"WS {o['WS']:.4f} personal {o['personal_aging']:+.4f} "
                  f"pooled {o['pooled_aging']:+.4f}", flush=True)
    return out


# ---------------- C: count-matched control, leaky vs fixed ----------------
def analysis_C(bl, reps=REPS):
    mice = sorted(bl)
    leaky = {"self": [], "one": [], "many": []}
    fixed = {"self": [], "one": [], "many": []}
    for m in mice:
        edays, elist, (Xl, yl) = bl[m]
        others, Xall, yall = others_early(bl, m)
        Xe = np.vstack([x for x, _ in elist]); ye = np.concatenate([y for _, y in elist])

        # ---- LEAKY: self scored on its OWN training data, no CV, while one/many are held out ----
        pc = _fit(Xe, ye)
        leaky["self"].append(_auc(pc, Xe, ye) - _auc(pc, Xl, yl))
        n = len(ye)
        ma, oa = [], []
        for rep in range(reps):
            rng = np.random.default_rng(7000 + 13 * rep + _seed(m))
            idx = rng.choice(len(yall), min(n, len(yall)), replace=False)
            if _both(yall[idx]):
                c = _fit(Xall[idx], yall[idx])
                ma.append(_auc(c, Xe, ye) - _auc(c, Xl, yl))
            o = str(rng.choice(others))
            Xo = np.vstack([x for x, _ in bl[o][1]]); yo = np.concatenate([y for _, y in bl[o][1]])
            if len(yo) >= n:
                j = rng.choice(len(yo), n, replace=False); Xo, yo = Xo[j], yo[j]
            if _both(yo):
                c = _fit(Xo, yo)
                oa.append(_auc(c, Xe, ye) - _auc(c, Xl, yl))
        if ma: leaky["many"].append(float(np.mean(ma)))
        if oa: leaky["one"].append(float(np.mean(oa)))

        # ---- FIXED: every arm trains on n events, scored on the SAME held-out session ----
        f_self, f_one, f_many = [], [], []
        for i in range(len(elist)):
            tr = [elist[j] for j in range(len(elist)) if j != i]
            Xt = np.vstack([x for x, _ in tr]); yt = np.concatenate([y for _, y in tr])
            Xh, yh = elist[i]
            if not _both(yt) or not _both(yh):
                continue
            n_s = len(yt)                                  # the self arm's natural budget
            c = _fit(Xt, yt)
            f_self.append(_auc(c, Xh, yh) - _auc(c, Xl, yl))
            mm, oo = [], []
            for rep in range(reps):
                rng = np.random.default_rng(9000 + 13 * rep + 7 * i + _seed(m))
                k = min(n_s, len(yall))
                idx = rng.choice(len(yall), k, replace=False)
                if _both(yall[idx]):
                    cm = _fit(Xall[idx], yall[idx])
                    mm.append(_auc(cm, Xh, yh) - _auc(cm, Xl, yl))
                o = str(rng.choice(others))
                Xo = np.vstack([x for x, _ in bl[o][1]]); yo = np.concatenate([y for _, y in bl[o][1]])
                if len(yo) >= n_s:
                    j = rng.choice(len(yo), n_s, replace=False); Xo, yo = Xo[j], yo[j]
                if _both(yo):
                    co = _fit(Xo, yo)
                    oo.append(_auc(co, Xh, yh) - _auc(co, Xl, yl))
            if mm: f_many.append(float(np.mean(mm)))
            if oo: f_one.append(float(np.mean(oo)))
        if f_self: fixed["self"].append(float(np.mean(f_self)))
        if f_many: fixed["many"].append(float(np.mean(f_many)))
        if f_one:  fixed["one"].append(float(np.mean(f_one)))

    def pack(d):
        return {k: {"aging": float(np.mean(v)), "n": len(v),
                    "sem": float(np.std(v) / max(1, np.sqrt(len(v))))} for k, v in d.items() if v}

    out = {"leaky": pack(leaky), "fixed": pack(fixed)}
    # paired contrasts on the corrected arms: is self ageing MORE than a count-matched other?
    ns = min(len(fixed["self"]), len(fixed["one"]), len(fixed["many"]))
    if ns:
        s = np.array(fixed["self"][:ns])
        out["fixed_contrasts"] = {
            "self_minus_one": _sign_test(s - np.array(fixed["one"][:ns]), seed=11),
            "self_minus_many": _sign_test(s - np.array(fixed["many"][:ns]), seed=12),
            "many_minus_one": _sign_test(np.array(fixed["many"][:ns]) - np.array(fixed["one"][:ns]),
                                         seed=13)}
    for k in ("leaky", "fixed"):
        o = out[k]
        print(f"  [C {k:5s}] self {o['self']['aging']:+.4f}  one {o['one']['aging']:+.4f}  "
              f"many {o['many']['aging']:+.4f}", flush=True)
    if ns:
        for k, v in out["fixed_contrasts"].items():
            print(f"    {k:16s} {v['mean']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] "
                  f"p={v['p_signflip']:.4f}  {v['frac_positive']:.0%} of {v['n']}", flush=True)
    return out


# ---------------- D: inclusion-rule sensitivity ----------------
def analysis_D(data, thresholds=(3, 4, 5, 6)):
    out = {}
    for k in thresholds:
        bl = blocks(data, min_sessions=k)
        if len(bl) < 4:
            continue
        a = analysis_A(bl, verbose=False)["fixed"]
        out[int(k)] = {"n_mice": a["n"], "asymmetry": a["mean"], "ci95": a["ci95"],
                       "p_signflip": a["p_signflip"], "frac_positive": a["frac_positive"],
                       "personal_aging": a["personal_aging"], "pooled_aging": a["pooled_aging"]}
        print(f"  [D >={k} sessions] {a['n']:2d} mice  asym {a['mean']:+.4f} "
              f"[{a['ci95'][0]:+.4f},{a['ci95'][1]:+.4f}] p={a['p_signflip']:.4f} "
              f"{a['frac_positive']:.0%}", flush=True)
    return out


# ---------------- E: why leaky and fixed differ here (decomposition) ----------------
def analysis_E(bl):
    """On BraiDyn the trial split INFLATED the asymmetry; here it deflates it. Two things differ
    between the protocols and only one of them is leakage, so measure them separately rather than
    attributing the reversal.

    Per mouse, four estimates of the personal decoder's early-block accuracy on the SAME trials:
      fixed        leave-one-SESSION-out, AUC computed per held-out session then averaged
      logo_pooled  leave-one-SESSION-out, out-of-fold predictions POOLED into a single AUC
      shuf_pooled  trials shuffled across sessions, out-of-fold predictions pooled into one AUC
      within       trained on 80% of a session and scored on the other 20% OF THAT SESSION
    giving
      leak_at_pooled   = shuf_pooled - logo_pooled   leakage, aggregation held fixed
      aggregation      = logo_pooled - fixed         aggregation, leakage held at zero
      within_session   = within - fixed              NOT a clean leak estimate: 80% of one session
                                                     is also far less training data than every other
                                                     session combined, so the two effects oppose
                                                     within it. Reported for completeness only;
                                                     leak_at_pooled is the matched comparison.
    A single pooled AUC across sessions is depressed by between-session shifts in the decision
    variable that a per-session AUC never sees, so the two differences can have opposite signs.
    The pooled arm is scored the same two ways for the same reason.
    """
    from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, cross_val_predict
    rows = []
    for m in sorted(bl):
        edays, elist, (Xl, yl) = bl[m]
        _, Xo, yo = others_early(bl, m)
        if not _both(yo):
            continue
        clf_pool = _fit(Xo, yo)
        Xe = np.vstack([x for x, _ in elist]); ye = np.concatenate([y for _, y in elist])
        g = np.concatenate([np.full(len(y), i) for i, (_, y) in enumerate(elist)])

        ws_fix, ls_fix, wthin = [], [], []
        for i, (Xh, yh) in enumerate(elist):
            tr = [elist[j] for j in range(len(elist)) if j != i]
            Xt = np.vstack([x for x, _ in tr]); yt = np.concatenate([y for _, y in tr])
            if not _both(yt) or not _both(yh):
                continue
            ws_fix.append(_auc(_fit(Xt, yt), Xh, yh))
            ls_fix.append(_auc(clf_pool, Xh, yh))
            k = min(5, int(min(np.sum(yh == c) for c in set(yh.tolist()))))
            if k >= 2:                                  # within-session CV: the leak, isolated
                pr = cross_val_predict(pilot.pipe(), Xh, yh, method="predict_proba",
                                       cv=StratifiedKFold(k, shuffle=True, random_state=0))[:, 1]
                from sklearn.metrics import roc_auc_score
                wthin.append(float(roc_auc_score(yh, pr)))
        if not ws_fix:
            continue
        from sklearn.metrics import roc_auc_score
        p_logo = cross_val_predict(pilot.pipe(), Xe, ye, groups=g, method="predict_proba",
                                   cv=LeaveOneGroupOut())[:, 1]
        p_shuf = cross_val_predict(pilot.pipe(), Xe, ye, method="predict_proba",
                                   cv=StratifiedKFold(5, shuffle=True, random_state=0))[:, 1]
        rows.append({"mouse": m,
                     "ws_fixed": float(np.mean(ws_fix)),
                     "ws_logo_pooled": float(roc_auc_score(ye, p_logo)),
                     "ws_shuf_pooled": float(roc_auc_score(ye, p_shuf)),
                     "ws_within": float(np.mean(wthin)) if wthin else None,
                     "ls_fixed": float(np.mean(ls_fix)),
                     "ls_pooled": _auc(clf_pool, Xe, ye)})

    def d(a, b):
        v = [r[a] - r[b] for r in rows if r[a] is not None and r[b] is not None]
        return _sign_test(v, seed=17)
    out = {"per_mouse": rows,
           "leak_at_pooled": d("ws_shuf_pooled", "ws_logo_pooled"),
           "aggregation": d("ws_logo_pooled", "ws_fixed"),
           "within_session_leak": d("ws_within", "ws_fixed"),
           "pooled_arm_aggregation": d("ls_pooled", "ls_fixed")}
    for k, v in out.items():
        if k == "per_mouse":
            continue
        print(f"  [E {k:22s}] {v['mean']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] "
              f"p={v['p_signflip']:.4f}  {v['frac_positive']:.0%} of {v['n']}", flush=True)
    return out


def main():
    g = gather()
    data, drops = g["per"], g["drops"]
    sess = {m: len(v) for m, v in data.items()}
    print(f"\nmice with usable sessions: {len(sess)}   sessions {sum(sess.values())}   "
          f"dropped experiments {len(drops)}", flush=True)
    for d in drops:
        print(f"  dropped {d['mouse']} {d['eid']}: {d['reason']}", flush=True)

    bl = blocks(data)
    print(f"\nanalysis cohort (>= {MIN_SESSIONS} sessions, early+late blocks): {len(bl)} mice", flush=True)

    print("\n=== A: aging asymmetry (leaky trial-split vs fixed session-held-out) ===", flush=True)
    A = analysis_A(bl)
    print("\n=== C: count-matched control (leaky vs fixed) ===", flush=True)
    C = analysis_C(bl)
    print("\n=== D: inclusion-rule sensitivity ===", flush=True)
    D = analysis_D(data)
    print("\n=== E: leakage vs score-aggregation decomposition ===", flush=True)
    E = analysis_E(bl)

    infl = A["leaky"]["mean"] - A["fixed"]["mean"]
    print(f"\ninflation from trial-split: {A['leaky']['mean']:+.4f} -> {A['fixed']['mean']:+.4f} "
          f"({infl:+.4f}, {100*infl/max(1e-9,A['leaky']['mean']):.0f}% of the leaky estimate)", flush=True)

    dest = os.path.join(HERE, "allen2p_leakfix.json")
    json.dump({"cohort": {"n_mice_usable": len(sess), "sessions_per_mouse": sess,
                          "dropped_experiments": drops, "n_mice_analysed": len(bl),
                          "min_sessions": MIN_SESSIONS},
               "A": A, "C": C, "D": D, "E": E, "trial_split_inflation": infl},
              open(dest, "w"), indent=1)
    print(f"\nsaved {dest}\nLEAKFIX2P_DONE", flush=True)


if __name__ == "__main__":
    main()
