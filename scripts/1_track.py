"""Step 1: detect + track players, save a video with boxes and the per-track crops.

Usage:
  .venv/bin/python scripts/1_track.py clips/bu_game.mp4 --name bu --start 600 --seconds 150
"""
import argparse, json, os
from collections import defaultdict
import cv2
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("video")
p.add_argument("--name", required=True, help="short label, e.g. bu or rhs")
p.add_argument("--start", type=float, default=0, help="start second in the video")
p.add_argument("--seconds", type=float, default=150)
p.add_argument("--model", default="yolo11m.pt")
p.add_argument("--conf", type=float, default=0.25)
p.add_argument("--imgsz", type=int, default=1280, help="detector input size; bigger finds small far-away players")
p.add_argument("--crop-every", type=int, default=5, help="save a crop every N frames per track")
a = p.parse_args()

out = f"out/{a.name}"
os.makedirs(f"{out}/crops", exist_ok=True)

cap = cv2.VideoCapture(a.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
W, H = int(cap.get(3)), int(cap.get(4))
cap.set(cv2.CAP_PROP_POS_FRAMES, int(a.start * fps))
n_frames = int(a.seconds * fps)
print(f"{a.video}: {W}x{H} @ {fps:.1f} fps, processing {a.seconds}s = {n_frames} frames")

writer = cv2.VideoWriter(f"{out}/boxes.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
model = YOLO(a.model)

frames_seen = defaultdict(list)   # track id -> frame numbers
crops_saved = defaultdict(int)
for i in range(n_frames):
    ok, frame = cap.read()
    if not ok:
        break
    # class 0 = person. persist=True keeps track ids across frames. ByteTrack is the tracker.
    r = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0],
                    conf=a.conf, imgsz=a.imgsz, device="mps", verbose=False)[0]
    vis = frame.copy()
    if r.boxes.id is not None:
        for box, tid in zip(r.boxes.xyxy.cpu().numpy().astype(int), r.boxes.id.cpu().numpy().astype(int)):
            x1, y1, x2, y2 = box
            frames_seen[int(tid)].append(i)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"#{tid}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if i % a.crop_every == 0 and (y2 - y1) >= 40:
                c = frame[max(0, y1):y2, max(0, x1):x2]
                if c.size:
                    cv2.imwrite(f"{out}/crops/t{tid:04d}_f{i:05d}.jpg", c)
                    crops_saved[int(tid)] += 1
    writer.write(vis)
    if i % 100 == 0:
        print(f"  frame {i}/{n_frames}", flush=True)

writer.release()
json.dump({"fps": fps, "tracks": frames_seen}, open(f"{out}/tracks.json", "w"))
long_tracks = [t for t, f in frames_seen.items() if len(f) >= fps]
print(f"Done. {len(frames_seen)} tracks total, {len(long_tracks)} lasted >= 1 second.")
print(f"Video with boxes: {out}/boxes.mp4   Crops: {out}/crops/")
