#!/usr/bin/env python3
"""
yolo_bench.py — run the CVZone YOLOv8 fake/real anti-spoof model as an
independent benchmark against amispoof, on the SAME scene.

This is the model from the CVZone / Murtaza Hassan "Anti-Spoofing" course
(models/l_version_1_300/l_version_1_300.pt — YOLOv8l, classes ["fake","real"]).
It is a RESEARCH REFERENCE only: a 2-class object detector trained on one
person's 640x480 webcam, known to overfit. We use it to cross-check amispoof's
verdict in a given lighting regime, NOT as a component.

Unlike the course's Main.py, this logs per-frame (verdict, confidence, fps) to a
CSV so the run is comparable to an amispoof session — no eyeballing.

Requires (pip, ~2.5 GB incl. torch):
    pip install ultralytics opencv-python

WEBCAM (live head-to-head, default — opens a window, press q to quit):
    python notebooks/yolo_bench.py --cam 0 --csv notebooks/data/yolo_live.csv

    # tip: the course used cam index 1 (external cam). A laptop's built-in
    # cam is usually 0. Try --cam 0 first, then --cam 1.

IMAGE FOLDER (offline, no camera — score a directory of frames):
    python notebooks/yolo_bench.py --images path/to/frames --csv out.csv

HEADLESS webcam (no window, e.g. over SSH — logs only, --seconds to bound it):
    python notebooks/yolo_bench.py --cam 0 --no-show --seconds 20 --csv out.csv
"""
import argparse
import csv
import glob
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "..", "models", "l_version_1_300", "l_version_1_300.pt")
CLASS_NAMES = ["fake", "real"]  # index 0 = fake, 1 = real (CVZone convention)


def load_model(model_path):
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit(
            "ultralytics not installed. Run:  pip install ultralytics opencv-python\n"
            "(pulls torch, ~2.5 GB — fine on unlimited wifi)"
        )
    if not os.path.exists(model_path):
        sys.exit(f"model not found: {model_path}")
    return YOLO(model_path)


IMGSZ = 640  # set by --imgsz; lower = faster on CPU
DEVICE = None  # set by --device: "0" = first GPU, "cpu", or None = auto-detect


def best_detection(result, conf_thresh):
    """Return (label, conf) of the highest-confidence box above threshold, or (None, 0)."""
    best_label, best_conf = None, 0.0
    for box in result.boxes:
        conf = float(box.conf[0])
        if conf > conf_thresh and conf > best_conf:
            cls = int(box.cls[0])
            best_label = CLASS_NAMES[cls] if 0 <= cls < len(CLASS_NAMES) else str(cls)
            best_conf = conf
    return best_label, best_conf


def run_webcam(model, cam, conf_thresh, show, seconds, writer, summary):
    import cv2

    cap = cv2.VideoCapture(cam)
    cap.set(3, 640)
    cap.set(4, 480)
    if not cap.isOpened():
        sys.exit(f"could not open camera index {cam} — try --cam 0 or --cam 1")

    prev = time.time()
    start = prev
    frame_idx = 0
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            results = model(img, stream=True, verbose=False, imgsz=IMGSZ, device=DEVICE)
            label, conf = None, 0.0
            for r in results:
                label, conf = best_detection(r, conf_thresh)
                if show:
                    _draw(cv2, img, r, conf_thresh)

            now = time.time()
            fps = 1.0 / (now - prev) if now != prev else 0.0
            prev = now
            _record(writer, summary, frame_idx, label, conf, fps)
            frame_idx += 1

            if show:
                cv2.putText(img, f"{(label or 'none').upper()} {int(conf*100)}%  {fps:.1f}fps",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("yolo_bench (q to quit)", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if seconds and (now - start) >= seconds:
                break
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()


def _draw(cv2, img, r, conf_thresh):
    for box in r.boxes:
        conf = float(box.conf[0])
        if conf <= conf_thresh:
            continue
        cls = int(box.cls[0])
        label = CLASS_NAMES[cls] if 0 <= cls < len(CLASS_NAMES) else str(cls)
        color = (0, 255, 0) if label == "real" else (0, 0, 255)
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)


def run_images(model, folder, conf_thresh, writer, summary):
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        paths.extend(glob.glob(os.path.join(folder, ext)))
    paths.sort()
    if not paths:
        sys.exit(f"no images found in {folder}")
    for i, p in enumerate(paths):
        results = model(p, verbose=False, imgsz=IMGSZ, device=DEVICE)
        label, conf = None, 0.0
        for r in results:
            label, conf = best_detection(r, conf_thresh)
        _record(writer, summary, i, label, conf, 0.0, name=os.path.basename(p))
        print(f"{os.path.basename(p):40s} -> {(label or 'none'):5s} {conf:.2f}")


def _record(writer, summary, idx, label, conf, fps, name=""):
    if writer:
        writer.writerow([idx, name, label or "none", f"{conf:.4f}", f"{fps:.2f}"])
    summary["frames"] += 1
    if label == "fake":
        summary["fake"] += 1
    elif label == "real":
        summary["real"] += 1
    else:
        summary["none"] += 1
    summary["fps_sum"] += fps


def main():
    ap = argparse.ArgumentParser(description="CVZone YOLOv8 fake/real benchmark for amispoof cross-check")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="path to l_version_1_300.pt")
    ap.add_argument("--cam", type=int, default=0, help="webcam index (0 built-in, 1 external)")
    ap.add_argument("--images", help="run on a folder of images instead of the webcam")
    ap.add_argument("--conf", type=float, default=0.6, help="confidence threshold")
    ap.add_argument("--imgsz", type=int, default=640, help="inference size (320/416 = faster on CPU)")
    ap.add_argument("--device", default=None, help='"0" = first GPU, "cpu", or omit for auto-detect')
    ap.add_argument("--no-show", dest="show", action="store_false", help="headless (no window)")
    ap.add_argument("--seconds", type=float, default=0, help="auto-stop webcam after N seconds (0 = until q)")
    ap.add_argument("--csv", help="write per-frame log here")
    args = ap.parse_args()

    global IMGSZ, DEVICE
    IMGSZ = args.imgsz
    DEVICE = args.device
    model = load_model(args.model)
    summary = {"frames": 0, "fake": 0, "real": 0, "none": 0, "fps_sum": 0.0}

    fh = writer = None
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        fh = open(args.csv, "w", newline="")
        writer = csv.writer(fh)
        writer.writerow(["frame", "name", "verdict", "confidence", "fps"])

    try:
        if args.images:
            run_images(model, args.images, args.conf, writer, summary)
        else:
            run_webcam(model, args.cam, args.conf, args.show, args.seconds, writer, summary)
    finally:
        if fh:
            fh.close()

    n = max(1, summary["frames"])
    print("\n=== yolo_bench summary ===")
    print(f"frames     : {summary['frames']}")
    print(f"fake       : {summary['fake']:5d}  ({100*summary['fake']/n:.1f}%)")
    print(f"real       : {summary['real']:5d}  ({100*summary['real']/n:.1f}%)")
    print(f"no-face    : {summary['none']:5d}  ({100*summary['none']/n:.1f}%)")
    if not args.images:
        print(f"avg fps    : {summary['fps_sum']/n:.1f}")
    if args.csv:
        print(f"csv        : {args.csv}")
    verdict = "fake" if summary["fake"] >= summary["real"] else "real"
    print(f"MAJORITY   : {verdict.upper()}")


if __name__ == "__main__":
    main()
