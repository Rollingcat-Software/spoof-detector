#!/usr/bin/env python3
"""
v3_check.py — summarize one amispoof session JSON against the V3 texture-collapse
veto logic. No deps beyond the stdlib.

    python notebooks/v3_check.py                  # newest *.json in notebooks/data
    python notebooks/v3_check.py path/to/file.json

Prints: environment, verdict, incident breakdown, and the two V3 signals
(texture.details.texture_score, screen_replay.details.skin_score) with the
exact fractions the veto checks (texture_score < 25 ; skin_score >= 30).
"""
import glob
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# V3 constants (mirror SessionEngine.checkTextureCollapseReplay)
TEXTURE_SPOOF_THRESHOLD = 25
TEXTURE_LOW_FRACTION = 0.30
COSIGNAL_SKIN_MIN = 30


def pick_file(argv):
    if len(argv) > 1:
        return argv[1]
    files = glob.glob(os.path.join(DATA, "*.json"))
    if not files:
        sys.exit(f"no JSON in {DATA}")
    return max(files, key=os.path.getmtime)


def series(frame_log, *path):
    out = []
    for f in frame_log:
        cur = f.get("analyzer_scores") or {}
        try:
            for p in path:
                cur = cur[p]
            if isinstance(cur, (int, float)):
                out.append(cur)
        except Exception:
            pass
    return out


def stat_line(name, xs, extra=""):
    if not xs:
        print(f"  {name:34s}: (none)")
        return
    print(f"  {name:34s}: n={len(xs)} min={min(xs):.1f} med={st.median(xs):.1f} "
          f"mean={sum(xs)/len(xs):.1f} max={max(xs):.1f}  {extra}")


def main():
    path = pick_file(sys.argv)
    d = json.load(open(path, encoding="utf-8"))
    env = d.get("environment", {})
    v = d.get("verdict", {})
    fl = d.get("frame_log", [])

    CURRENT_BUILDS = ("threat-coop", "prod-cdn-restore")
    ver = d.get("amispoof_version") or "unknown"
    build_tag = "CURRENT" if any(b in ver for b in CURRENT_BUILDS) else "STALE — verdict from obsolete logic"
    print(f"=== {os.path.basename(path)} ===")
    print(f"  build={ver}  [{build_tag}]")
    print(f"  label={env.get('capture_label')}  ambient={env.get('ambient_label')}  "
          f"device={env.get('replay_device')}  notes={env.get('notes')}")
    print(f"  VERDICT: is_live={v.get('is_live')}  conf={v.get('confidence')}  "
          f"threat={v.get('dominant_threat')}  fps={d.get('fps_smoothed')}")
    print(f"  {v.get('session_duration_sec')}s  {v.get('frames_analyzed')} frames  "
          f"blinks={v.get('blink_count')}  quality_uncertain={v.get('quality_uncertain')}")
    cs = v.get("category_scores") or {}
    print("  category_scores:", {k: round(x, 3) for k, x in cs.items()})

    inc = v.get("incidents") or []
    from collections import Counter
    cats = Counter((i.get("category") if isinstance(i, dict) else i) for i in inc)
    print(f"  INCIDENTS ({len(inc)}):", dict(cats))
    for i in inc[:8]:
        if isinstance(i, dict):
            print(f"    - t={i.get('timestamp')} {i.get('category')}: {i.get('description')}")

    tex = series(fl, "texture", "details", "texture_score")
    skin = series(fl, "screen_replay", "details", "skin_score")
    print("=== V3 SIGNALS ===")
    tfrac = sum(1 for x in tex if x < TEXTURE_SPOOF_THRESHOLD) / len(tex) if tex else 0
    sfrac = sum(1 for x in skin if x >= COSIGNAL_SKIN_MIN) / len(skin) if skin else 0
    stat_line("texture.texture_score", tex,
              f"<25 in {tfrac:.0%} of frames (veto needs >=30% in a 30-frame window)")
    stat_line("screen_replay.skin_score", skin,
              f">=30 in {sfrac:.0%} (co-signal: median-in-window >=30)")
    smed = st.median(skin) if skin else 0
    print("=== READING ===")
    if tex and skin:
        texture_collapsing = tfrac >= TEXTURE_LOW_FRACTION
        cosignal = smed >= COSIGNAL_SKIN_MIN
        if texture_collapsing and cosignal:
            print("  -> V3 WOULD FIRE: texture collapsed AND skin_score median >=30 (REPLAY-like)")
        elif texture_collapsing and not cosignal:
            print(f"  -> SPARED: texture collapsed but skin_score median {smed:.1f} < 30 (LIVE-like)")
        elif not texture_collapsing:
            print(f"  -> QUIET: texture did not broadly collapse (only {tfrac:.0%} < 25)")


if __name__ == "__main__":
    main()
