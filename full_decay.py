"""Full 15-day decay regression (DANDI:001425): personal vs pooled decoder accuracy over ALL days.

The block analysis compares early (1-5) vs late (11-15) blocks. The earlier slope analysis dropped
days 1-5 to avoid leakage, which under-counts the effect. This does it properly over the whole
protocol: for every session day we get the accuracy of the personal decoder and the pooled decoder,
regress each on day, and regress the personal-minus-pooled GAP on day. A negative gap-slope means
the personal decoder's early advantage narrows over the 15 days as it drifts.

  personal accuracy on day d:
    d in early block (<=5): leave-one-early-session-out -- train on the OTHER early sessions, test d
                            (no leakage; the decoder never trains on the session it is tested on)
    d after early block:    train on all early sessions, test d
  pooled accuracy on day d: train on the OTHER mice's early sessions, test M's day d (never leaks)

Per mouse we fit slope_personal, slope_pooled, and slope_gap (AUC/day). Paired sign-flip tests over
mice. Also emits per-day mean AUC curves. Streams parcellated dF/F only. Saves full_decay.json.
"""
import re, json, os, socket
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

socket.setdefaulttimeout(90)
TARGETS = ["state_lever", "lick", "reward", "tone"]
EARLY_MAX = 5
MIN_DAYS = 4            # need >=4 test days across the protocol to fit a slope
HERE = os.path.dirname(os.path.abspath(__file__))


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


def pipe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))


def _both(y):
    return len(set(y.tolist())) == 2


def _pool(sess, days):
    xs = [sess[d][0] for d in days if d in sess]; ys = [sess[d][1] for d in days if d in sess]
    if not xs:
        return None
    return np.vstack(xs), np.concatenate(ys)


def analyze(target, per_mouse):
    from sklearn.metrics import roc_auc_score
    early = {}
    for m in per_mouse:
        e = _pool(per_mouse[m], [d for d in per_mouse[m] if d <= EARLY_MAX])
        if e is not None and _both(e[1]):
            early[m] = e
    mice = sorted(early)

    rows = []                       # (mouse, slope_personal, slope_pooled, slope_gap, n_days)
    cur_p = defaultdict(list); cur_l = defaultdict(list); cur_g = defaultdict(list)
    for m in mice:
        sess = per_mouse[m]
        early_days = [d for d in sess if d <= EARLY_MAX]
        others = [o for o in mice if o != m]
        Xtr_o = np.vstack([early[o][0] for o in others]); ytr_o = np.concatenate([early[o][1] for o in others])
        if not _both(ytr_o):
            continue
        clf_pool = pipe().fit(Xtr_o, ytr_o)     # pooled decoder (fixed for this mouse)

        days, ap, al = [], [], []
        for d in sorted(sess):
            Xd, yd = sess[d]
            if not _both(yd):
                continue
            # pooled accuracy on day d (never leaks)
            a_l = float(roc_auc_score(yd, clf_pool.predict_proba(Xd)[:, 1]))
            # personal accuracy on day d
            if d <= EARLY_MAX:
                tr_days = [e for e in early_days if e != d]           # leave-this-early-session-out
                pooled = _pool(sess, tr_days)
                if pooled is None or not _both(pooled[1]):
                    continue
                clf_p = pipe().fit(pooled[0], pooled[1])
            else:
                clf_p = pipe().fit(early[m][0], early[m][1])          # train on all early
            a_p = float(roc_auc_score(yd, clf_p.predict_proba(Xd)[:, 1]))
            days.append(d); ap.append(a_p); al.append(a_l)
            cur_p[d].append(a_p); cur_l[d].append(a_l); cur_g[d].append(a_p - a_l)
        if len(days) >= MIN_DAYS:
            sp = float(np.polyfit(days, ap, 1)[0])
            sl = float(np.polyfit(days, al, 1)[0])
            sg = float(np.polyfit(days, np.array(ap) - np.array(al), 1)[0])
            rows.append((m, sp, sl, sg, len(days)))

    def paired(vals):
        v = np.array(vals); rng = np.random.default_rng(0)
        null = np.array([(v * rng.choice([-1, 1], len(v))).mean() for _ in range(20000)])
        return float(v.mean()), float((1 + np.sum(np.abs(null) >= abs(v.mean()))) / (1 + len(null)))

    sp_m, _ = paired([r[1] for r in rows])
    sl_m, _ = paired([r[2] for r in rows])
    sg_m, sg_p = paired([r[3] for r in rows])
    curve = {int(d): {"personal": float(np.mean(cur_p[d])), "pooled": float(np.mean(cur_l[d])),
                      "gap": float(np.mean(cur_g[d])), "n": len(cur_p[d])} for d in sorted(cur_p)}
    return {"target": target, "n_mice": len(rows),
            "slope_personal_per_day": sp_m, "slope_pooled_per_day": sl_m,
            "slope_gap_per_day": sg_m, "gap_slope_p": sg_p,
            "frac_gap_narrows": float(np.mean([r[3] < 0 for r in rows])),
            "decay_curve": curve,
            "per_mouse": {r[0]: {"sp": round(r[1], 5), "sl": round(r[2], 5),
                                 "sgap": round(r[3], 5), "n": r[4]} for r in rows}}


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
        pm = {m: dict(v) for m, v in data[tgt].items() if len(v) >= 3}
        r = analyze(tgt, pm)
        out["targets"][tgt] = r
        print(f"  [{tgt}] personal {r['slope_personal_per_day']:+.5f}/d  pooled {r['slope_pooled_per_day']:+.5f}/d  "
              f"GAP {r['slope_gap_per_day']:+.5f}/d p={r['gap_slope_p']:.4f} "
              f"(gap narrows in {r['frac_gap_narrows']*100:.0f}%, n={r['n_mice']})", flush=True)
    dest = os.path.join(HERE, "full_decay.json")
    json.dump(out, open(dest, "w"), indent=1)
    print(f"\nsaved {dest}\nFULL_DECAY_DONE", flush=True)


if __name__ == "__main__":
    main()
