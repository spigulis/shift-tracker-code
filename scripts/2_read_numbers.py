"""Step 2: try to read a jersey number from each track's crops, then vote.

Usage:
  .venv/bin/python scripts/2_read_numbers.py --name bu
"""
import argparse, glob, os, re, json
from collections import defaultdict, Counter
import cv2
import easyocr

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--min-conf", type=float, default=0.5, help="a single read counts only above this")
p.add_argument("--min-votes", type=int, default=2, help="a track is 'confident' if the winning number got this many reads")
p.add_argument("--team", choices=["red", "any"], default="any", help="only read tracks whose jersey looks this color")
p.add_argument("--max-crops", type=int, default=15, help="max crops read per track (evenly spaced)")
a = p.parse_args()

out = f"out/{a.name}"
reader = easyocr.Reader(["en"], gpu=False)  # EasyOCR's GPU path doesn't use Apple MPS well; fine for short clips

def prep(img):
    # keep the upper body (back/chest numbers), upscale, boost contrast
    h = img.shape[0]
    img = img[int(h * 0.1):int(h * 0.65)]
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(2.0, (4, 4)).apply(g)

def is_red(img):
    h, w = img.shape[:2]
    t = img[int(h * .15):int(h * .55), int(w * .25):int(w * .75)]
    hsv = cv2.cvtColor(t, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 0] < 10) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 100) & (hsv[..., 2] > 60)
    return m.mean() > 0.25

tracks = json.load(open(f"{out}/tracks.json"))
fps = tracks["fps"]
long_ids = sorted(int(t) for t, fr in tracks["tracks"].items() if len(fr) >= fps)

by_track = defaultdict(list)
for f in sorted(glob.glob(f"{out}/crops/*.jpg")):
    t = int(os.path.basename(f)[1:5])
    if t in long_ids:
        by_track[t].append(f)
files, kept_ids = [], []
for t, fl in by_track.items():
    if a.team == "red":
        votes_red = [is_red(cv2.imread(f)) for f in fl]
        if sum(votes_red) < 0.5 * len(fl):
            continue
    kept_ids.append(t)
    step = max(1, len(fl) // a.max_crops)
    files += fl[::step][:a.max_crops]
long_ids = sorted(kept_ids)
reads = defaultdict(list)   # track -> [(number, conf, frame)]
print(f"Reading {len(files)} crops...")
for k, f in enumerate(files):
    tid = int(os.path.basename(f)[1:5])
    img = cv2.imread(f)
    for _, text, conf in reader.readtext(prep(img), allowlist="0123456789", detail=1):
        if re.fullmatch(r"\d{1,2}", text) and conf >= a.min_conf:
            reads[tid].append((str(int(text)), float(conf), int(os.path.basename(f)[7:12])))
    if k % 50 == 0:
        print(f"  {k}/{len(files)}", flush=True)

rows, confident = [], 0
for t in long_ids:
    votes = Counter(n for n, _, _ in reads.get(t, []))
    if votes:
        num, cnt = votes.most_common(1)[0]
        ok = cnt >= a.min_votes
    else:
        num, cnt, ok = "", 0, False
    confident += ok
    fr = tracks["tracks"][str(t)]
    rows.append((t, f"{fr[0]/fps:.0f}-{fr[-1]/fps:.0f}s", num, cnt, sum(votes.values()), ok))

with open(f"{out}/numbers.csv", "w") as fh:
    fh.write("track,seconds_in_clip,number,votes_for_it,total_reads,confident,correct_by_eye\n")
    for r in rows:
        fh.write(",".join(map(str, r)) + ",\n")

print(f"\nTracks lasting >= 1s (after team filter): {len(long_ids)}")
print(f"Confident number read: {confident}")
print(f"Fill in 'correct_by_eye' in {out}/numbers.csv (y/n) after watching boxes.mp4.")
for r in rows:
    print(f"  track {r[0]:>4}  seen {r[1]:>9}  number {r[2] or '-':>3}  votes {r[3]}/{r[4]}  {'CONFIDENT' if r[5] else ''}")
