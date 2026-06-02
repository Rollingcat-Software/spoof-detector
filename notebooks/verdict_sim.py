#!/usr/bin/env python3
"""Faithful-enough END-TO-END verdict + confidence replay over the labelled
current-build sessions, to (1) verify the shipped fixes net-improve verdict
accuracy and (2) test confidence-formula candidates.

Fidelity: the fuser REAL prob is the weight-normalized mean of analyzer scores
(every SPOOF_SIGNAL_MAP row sums to ~1); the SessionEngine then forms
  avgReal      = mean of per-frame REAL prob
  worstWindow  = min over 5-frame windows of REAL prob   (peak-sensitive)
  blendedReal  = 0.5*avg + 0.5*worst   (== adjustedReal at dataConfidence=1,
                 ignoring the small temporal/incident terms, which apply to OLD
                 and NEW equally). The prover term uses the RECORDED proof_total
                 (frame_log) since the prover needs raw landmarks we don't store.
Verdict model (isolates the shipped fixes; prover/quality held as pass, which
they are for these proven-live sessions and unchanged by the fixes):
  is_live = incidents < 3  AND  blendedReal > 0.45

OLD config = original DEFAULT weights + ungated texture veto.
NEW config = shipped reliability weights + motion-blur-gated veto (PR #96).

Run:  python notebooks/verdict_sim.py
"""
from __future__ import annotations
import glob, json, statistics as st
from collections import deque

CURRENT = ("threat-coop", "prod-cdn-restore", "motion-typing")
NYQUIST = {"screen_flicker": 18, "micro_tremor": 20, "rppg": 10}
DISABLED = {"temporal", "screen_flicker", "micro_tremor", "expression_dynamics",
            "background_grid"}
WEIGHTS_OLD = {
    "minifasnet": 3.0, "planarity": 2.0, "screen_flicker": 3.0,
    "landmark_variance": 2.0, "background_grid": 1.5, "device_boundary": 0.5,
    "micro_tremor": 0.5, "rppg": 0.5, "blink": 0.5, "screen_replay": 0.5,
    "ar_filter": 0.3, "temporal": 0.3, "texture": 1.5, "moire": 0.0,
}
WEIGHTS_NEW = {
    "minifasnet": 2.0, "gaze": 2.0, "blink_symmetry": 2.0, "blink": 2.0,
    "behavioral_pattern": 1.5, "device_boundary": 1.5, "planarity": 0.5,
    "texture": 0.5, "landmark_variance": 0.5, "eyebrow_motion": 0.3,
    "ar_filter": 0.3, "rppg": 0.0, "screen_replay": 0.0,
    "pose_3d_consistency": 0.0, "moire": 0.0, "background_motion": 0.0,
    "screen_flicker": 0.0, "micro_tremor": 0.0, "temporal": 0.0,
    "background_grid": 0.0, "expression_dynamics": 0.0, "hand_tracking": 0.0,
    "voice_activity": 0.0, "audio_mouth_sync": 0.0,
}
TEX_THR = 25; LOWFRAC = 0.30; MIN_EL = 3.0; THROT = 2.5; WMIN = 20; WTEX = 30
WRIG = 150; SKIN_MIN = 30; SKIN_MAX = 5; STILL_MAX = 30; MIN_STILL = 5


def col(l):
    l = (l or "U").upper()
    return "LIVE" if l == "LIVE" else (
        "SPOOF" if l.startswith(("REPLAY", "SCREEN", "PRINT", "MASK", "DEEPFAKE")) else None)


def sub(fr, an, key):
    a = (fr.get("analyzer_scores") or {}).get(an)
    if not isinstance(a, dict):
        return None
    x = (a.get("details") or {}).get(key)
    return x if isinstance(x, (int, float)) else None


def topscore(fr, an):
    a = (fr.get("analyzer_scores") or {}).get(an)
    if not isinstance(a, dict):
        return None
    s = a.get("score")
    return s if isinstance(s, (int, float)) else None


def frame_realprob(fr, weights):
    asc = fr.get("analyzer_scores") or {}
    fps = fr.get("fps")
    num = den = 0.0
    for name in asc:
        if name in DISABLED:
            continue
        w = weights.get(name, 0.5)  # fuser ?? 0.5 fallback
        if w <= 0:
            continue
        if name in NYQUIST and isinstance(fps, (int, float)) and 0 < fps < NYQUIST[name]:
            continue
        s = topscore(fr, name)
        if s is None:
            continue
        num += w * (s / 100.0); den += w
    return (num / den) if den > 0 else None


def veto_incidents(fl, gated):
    rt = deque(maxlen=WTEX); rs = deque(maxlen=WTEX); rr = deque(maxlen=WRIG)
    last = -999; inc = 0
    for fr in fl:
        t = sub(fr, "texture", "texture_score"); sk = sub(fr, "screen_replay", "skin_score")
        rv = sub(fr, "landmark_variance", "overall_var")
        if t is None:
            continue
        rt.append(t); rs.append(sk if sk is not None else float("nan"))
        rr.append(rv if rv is not None else float("nan"))
        el = fr.get("t_sec") or 0
        if el < MIN_EL or len(rt) < WMIN or el - last < THROT:
            continue
        samples = list(rt)
        if sum(1 for s in samples if s < TEX_THR) / len(samples) < LOWFRAC:
            continue
        if gated:  # motion-blur suppressor (end-aligned)
            rig = list(rr); n = min(len(samples), len(rig)); sl = stt = 0
            for k in range(1, n + 1):
                rg = rig[-k]
                if rg != rg or rg >= STILL_MAX:
                    continue
                stt += 1
                if samples[-k] < TEX_THR:
                    sl += 1
            if stt >= MIN_STILL and sl / stt < LOWFRAC:
                continue
        sk_s = [x for x in rs if x == x]
        if len(sk_s) < WMIN / 2:
            continue
        sk_med = sorted(sk_s)[len(sk_s) // 2]
        isScreen = sk_med >= SKIN_MIN; isPhoto = sk_med < SKIN_MAX
        if gated and isPhoto and not isScreen:
            rig = [x for x in rr if x == x]
            if rig and sorted(rig)[len(rig) // 2] >= STILL_MAX:
                isPhoto = False
        if not isScreen and not isPhoto:
            continue
        inc += 1; last = el
    return inc


def blended_real(fl, weights):
    rp = [frame_realprob(fr, weights) for fr in fl]
    rp = [x for x in rp if x is not None]
    if not rp:
        return 0.5
    avg = sum(rp) / len(rp)
    worst = avg
    if len(rp) >= 3:
        w = min(5, len(rp))
        for i in range(len(rp) - w + 1):
            wa = sum(rp[i:i + w]) / w
            worst = min(worst, wa)
    return 0.5 * avg + 0.5 * worst


def conf_old(is_live, blended, incidents, proof):
    # current formula (dataConfidence=1): +0.3 floor + 0.3*activity + margin
    return min(1.0, 0.3 * proof + 0.3 + 0.4 * max(0.0, blended - 0.3))


def conf_new(is_live, blended, incidents, proof):
    # honest certainty: distance from the 0.45 decision boundary, no floor, proof
    # only a mild LIVE corroborator. SPOOF certainty also rises with incidents.
    if not is_live:
        c = max((0.45 - blended) / 0.45 if blended < 0.45 else 0.0, min(1.0, incidents / 6.0))
    else:
        c = (blended - 0.45) / (0.85 - 0.45) * (0.6 + 0.4 * proof)
    return max(0.0, min(1.0, c))


def main():
    sess = []
    for f in glob.glob("notebooks/data/*.json"):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if not any(b in (d.get("amispoof_version") or "") for b in CURRENT):
            continue
        c = col((d.get("environment") or {}).get("capture_label"))
        fl = d.get("frame_log") or []
        if c and fl:
            proof = 0.0
            for fr in reversed(fl):
                pt = fr.get("proof_total")
                if isinstance(pt, (int, float)):
                    proof = pt / 100.0; break
            sess.append((c, fl, proof))
    nL = sum(1 for c, _, _ in sess if c == "LIVE")
    print(f"sessions {len(sess)} (LIVE {nL}, SPOOF {len(sess)-nL})\n")

    print("=== (1) verdict accuracy: OLD fixes vs NEW fixes (end-to-end) ===")
    for tag, W, gated in (("OLD (orig weights+ungated veto)", WEIGHTS_OLD, False),
                          ("NEW (reliability+motion-gated)", WEIGHTS_NEW, True)):
        ok = fa = fr_ = 0
        for c, fl, proof in sess:
            br = blended_real(fl, W); inc = veto_incidents(fl, gated)
            is_live = inc < 3 and br > 0.45
            if c == "LIVE" and is_live or c == "SPOOF" and not is_live:
                ok += 1
            elif c == "SPOOF" and is_live:
                fa += 1
            else:
                fr_ += 1
        print(f"  {tag:34} acc {ok}/{len(sess)} = {100*ok/len(sess):.0f}%  "
              f"(false-accept {fa}, false-reject {fr_})")

    print("\n=== (2) confidence reliability under NEW fixes: does confidence track correctness? ===")
    rows = []
    for c, fl, proof in sess:
        br = blended_real(fl, WEIGHTS_NEW); inc = veto_incidents(fl, True)
        is_live = inc < 3 and br > 0.45
        correct = (c == "LIVE") == is_live
        rows.append((correct, conf_old(is_live, br, inc, proof),
                     conf_new(is_live, br, inc, proof)))
    for name, idx in (("OLD formula (+0.3 floor + activity)", 1), ("NEW formula (boundary margin)", 2)):
        cc = [r[idx] for r in rows if r[0]]
        ww = [r[idx] for r in rows if not r[0]]
        gap = (st.mean(cc) - st.mean(ww)) if cc and ww else float("nan")
        print(f"  {name:36} mean-conf correct {st.mean(cc):.2f} vs wrong {st.mean(ww):.2f}  "
              f"(separation {gap:+.2f})")


if __name__ == "__main__":
    main()
