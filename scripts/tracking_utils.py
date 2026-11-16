"""Small helper functions for tracking: ice mask, team colour, referee test, track merging.
No GPU needed here, so these can be tested anywhere."""
import cv2
import numpy as np


def _clip_box(box, shape):
    h, w = shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def on_ice(frame, box, min_frac=0.35):
    """True if the strip just below the feet looks like ice (bright, low colour).
    Bench players, spectators and people behind the glass have boards, seats or
    floor under their feet, so they fail this test. Works even when the camera moves."""
    x1, y1, x2, y2 = [int(v) for v in box]
    bh, bw = y2 - y1, x2 - x1
    sx1, sx2 = int(x1 + bw * 0.1), int(x2 - bw * 0.1)
    sy1, sy2 = y2 - int(bh * 0.05), y2 + max(4, int(bh * 0.2))
    sx1, sy1, sx2, sy2 = _clip_box((sx1, sy1, sx2, sy2), frame.shape)
    if sx2 - sx1 < 3 or sy2 - sy1 < 3:
        return False
    hsv = cv2.cvtColor(frame[sy1:sy2, sx1:sx2], cv2.COLOR_BGR2HSV)
    ice = (hsv[..., 1] < 60) & (hsv[..., 2] > 140)
    return float(ice.mean()) >= min_frac


def torso(frame, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    bh, bw = y2 - y1, x2 - x1
    return frame[max(0, y1 + int(bh * .15)):y1 + int(bh * .55),
                 max(0, x1 + int(bw * .25)):x1 + int(bw * .75)]


def red_fraction(frame, box):
    t = torso(frame, box)
    if t.size == 0:
        return 0.0
    hsv = cv2.cvtColor(t, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 0] < 10) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 100) & (hsv[..., 2] > 60)
    return float(m.mean())


def is_referee(frame, box):
    """Referees wear black and white stripes: lots of near-black AND lots of near-white
    in the torso, and almost no red."""
    t = torso(frame, box)
    if t.size == 0:
        return False
    hsv = cv2.cvtColor(t, cv2.COLOR_BGR2HSV)
    dark = (hsv[..., 2] < 70).mean()
    white = ((hsv[..., 1] < 50) & (hsv[..., 2] > 170)).mean()
    return dark > 0.25 and white > 0.2 and red_fraction(frame, box) < 0.1


def merge_tracks(tracks, fps, max_gap_s=3.0, base_px=80, px_per_s=150):
    """Join track fragments that are clearly one player.

    tracks: {id: [(frame, x1, y1, x2, y2), ...]} sorted by frame.
    Two fragments A -> B are joined if B starts after A ends, the gap is at most
    max_gap_s, B starts near where A ended (allowing more distance for longer gaps),
    and the box height is similar. Each fragment gets at most one successor and one
    predecessor. Returns {new_id: [rows...]} and {old_id: new_id}.
    """
    ids = sorted(tracks, key=lambda t: tracks[t][0][0])
    info = {}
    for t in ids:
        r = tracks[t]
        f0, f1 = r[0], r[-1]
        info[t] = dict(
            start=f0[0], end=f1[0],
            c0=((f0[1] + f0[3]) / 2, (f0[2] + f0[4]) / 2), h0=f0[4] - f0[2],
            c1=((f1[1] + f1[3]) / 2, (f1[2] + f1[4]) / 2), h1=f1[4] - f1[2])
    cands = []
    for a in ids:
        for b in ids:
            if a == b or info[b]["start"] <= info[a]["end"]:
                continue
            gap = (info[b]["start"] - info[a]["end"]) / fps
            if gap > max_gap_s:
                continue
            d = np.hypot(info[a]["c1"][0] - info[b]["c0"][0], info[a]["c1"][1] - info[b]["c0"][1])
            if d > base_px + px_per_s * gap:
                continue
            ratio = info[b]["h0"] / max(1, info[a]["h1"])
            if not 0.6 <= ratio <= 1.6:
                continue
            cands.append((d / (base_px + px_per_s * gap) + gap / max_gap_s, a, b))
    cands.sort()
    nxt, prv = {}, {}
    for _, a, b in cands:
        if a in nxt or b in prv:
            continue
        nxt[a], prv[b] = b, a
    new_id, merged = {}, {}
    for t in ids:
        if t in prv:
            continue
        chain, cur = [t], t
        while cur in nxt:
            cur = nxt[cur]
            chain.append(cur)
        rows = []
        for c in chain:
            rows += tracks[c]
            new_id[c] = t
        merged[t] = rows
    return merged, new_id
