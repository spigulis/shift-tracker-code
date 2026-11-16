"""Step 1 (Colab): detect + track players with YOLO + BoT-SORT/OSNet re-ID (BoxMOT).

Drops everything that is not on the ice and drops referees before tracking.
Tracks both teams (so IDs don't swap between teams), then keeps only red-jersey tracks,
merges broken fragments, and writes a boxes video + tracks.json.

Usage (in Colab):
  python scripts/track_boxmot.py VIDEO --out OUTDIR --start 720 --seconds 150
"""
import argparse, json, os, statistics, subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from boxmot import BotSort
from ultralytics import YOLO

from tracking_utils import on_ice, is_referee, red_fraction, merge_tracks

p = argparse.ArgumentParser()
p.add_argument("video")
p.add_argument("--out", required=True)
p.add_argument("--start", type=float, default=0)
p.add_argument("--seconds", type=float, default=150)
p.add_argument("--model", default="yolo11m.pt")
p.add_argument("--imgsz", type=int, default=1280)
p.add_argument("--reid", default="osnet_x1_0_msmt17.pt")
p.add_argument("--min-red", type=float, default=0.15, help="track is red if this share of its boxes look red")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cap = cv2.VideoCapture(a.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
W, H = int(cap.get(3)), int(cap.get(4))
cap.set(cv2.CAP_PROP_POS_FRAMES, int(a.start * fps))
n_frames = int(a.seconds * fps)
print(f"{a.video}: {W}x{H} @ {fps:.0f} fps, {a.seconds}s from {a.start}s, device {dev}")

yolo = YOLO(a.model)
tracker = BotSort(reid_weights=Path(a.reid), device=dev, half=dev.type == "cuda",
                  track_buffer=int(fps * 3), frame_rate=int(fps),
                  track_high_thresh=0.4, track_low_thresh=0.1, new_track_thresh=0.5)

rows = defaultdict(list)      # track id -> [(frame, x1, y1, x2, y2)]
red_votes = defaultdict(list)  # track id -> [red fraction per box]
ice_votes = defaultdict(list)  # track id -> [on-ice yes/no per box]
raw_dets = kept_dets = 0
for i in range(n_frames):
    ok, frame = cap.read()
    if not ok:
        break
    r = yolo.predict(frame, classes=[0], conf=0.1, imgsz=a.imgsz, device=dev, verbose=False)[0]
    d = r.boxes.data.cpu().numpy()          # x1 y1 x2 y2 conf cls
    raw_dets += len(d)
    # Soft per-box filter (so a skater near the glass isn't lost), strict ice share is judged per track later
    keep = [k for k in range(len(d))
            if (d[k, 3] - d[k, 1]) >= 30 and on_ice(frame, d[k, :4], 0.12) and not is_referee(frame, d[k, :4])]
    d = d[keep] if keep else np.empty((0, 6))
    kept_dets += len(d)
    strict = [on_ice(frame, d[k, :4], 0.35) for k in range(len(d))]
    out = tracker.update(d, frame)           # x1 y1 x2 y2 id conf cls det_ind
    for o in (out if len(out) else []):
        tid = int(o[4])
        box = o[:4]
        rows[tid].append((i, *[float(v) for v in box]))
        red_votes[tid].append(red_fraction(frame, box))
        ice_votes[tid].append(strict[int(o[7])] if 0 <= int(o[7]) < len(strict) else False)
    if i % 300 == 0:
        print(f"  frame {i}/{n_frames}", flush=True)

def stats(tr, label):
    lens = [len(v) / fps for v in tr.values()]
    long_ = [l for l in lens if l >= 1]
    print(f"{label}: {len(tr)} tracks, median {statistics.median(lens):.1f}s, "
          f"{len(long_)} lasting >= 1s, longest {max(lens):.0f}s")

print(f"\nDetections: {raw_dets} raw, {kept_dets} kept after ice + referee filter")
stats(rows, "All tracks (both teams, on ice)")
on_the_ice = {t: v for t, v in rows.items() if np.mean(ice_votes[t]) >= 0.5}
stats(on_the_ice, "Tracks mostly on the ice")
red = {t: v for t, v in on_the_ice.items() if np.mean([x > a.min_red for x in red_votes[t]]) >= 0.5}
stats(red, "Red (BU) tracks before merge")
merged, new_id = merge_tracks(red, fps)
stats(merged, "Red (BU) tracks after merge")

json.dump({"fps": fps, "start": a.start, "tracks": {str(k): v for k, v in merged.items()},
           "merge_map": {str(k): v for k, v in new_id.items()}}, open(f"{a.out}/tracks.json", "w"))

# boxes video of the final red tracks
cap.set(cv2.CAP_PROP_POS_FRAMES, int(a.start * fps))
by_frame = defaultdict(list)
for t, v in merged.items():
    for f, x1, y1, x2, y2 in v:
        by_frame[f].append((t, int(x1), int(y1), int(x2), int(y2)))
raw_path = f"{a.out}/boxes_raw.mp4"
wr = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
for i in range(n_frames):
    ok, frame = cap.read()
    if not ok:
        break
    for t, x1, y1, x2, y2 in by_frame.get(i, []):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"#{t}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    wr.write(frame)
wr.release()
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", raw_path, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "23", f"{a.out}/boxes.mp4"], check=True)
os.remove(raw_path)
print(f"\nDone. Video: {a.out}/boxes.mp4   Tracks: {a.out}/tracks.json")
