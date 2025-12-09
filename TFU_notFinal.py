import cv2
import subprocess
import numpy as np
import time
import os
from ultralytics import YOLO

# ---------------------------------------------------
# DISTANCE CALIBRATION (TUNE THIS!)
# ---------------------------------------------------
# Approximate vertical scale: how many pixels in the SMALL (320x320) frame
# correspond to 1 meter along the road direction.
#
# You MUST tune this using a known distance in your scene:
#   PIXELS_PER_METER = (pixel distance between two known markers) / (meters between them)
#
PIXELS_PER_METER = 5.0  # <-- placeholder, adjust after seeing real camera view

# Thresholds (meters) upstream of the stop line
THRESHOLDS = [30.0, 20.0, 10.0]

# ---------------------------------------------------
# General setup
# ---------------------------------------------------
SNAP_DIR = "snapshots"
os.makedirs(SNAP_DIR, exist_ok=True)

# rpicam-vid config
cmd = [
    "rpicam-vid",
    "--inline",
    "-t", "0",
    "--width", "640",
    "--height", "480",
    "--framerate", "10",
    "--codec", "mjpeg",
    "--quality", "10",
    "-o", "-",
]

proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0)

# Load ONNX model
model = YOLO("yolo11n.onnx")  # tiny model exported earlier

jpeg_data = b""

# ------------- SIMPLE TRACKER STATE ----------------
# For each car (track_id) we store:
#  - center (cx, cy)
#  - last_time  (for housekeeping if you want later)
#  - last_dist  (last distance to stop line in meters)
#  - hit_30, hit_20, hit_10 flags so we only trigger once
tracks = {}
next_track_id = 0
MAX_MATCH_DIST = 50  # pixels

# Stop line location in relative image coordinates
STOP_LINE_Y_REL = 0.7  # 70% down the 320px image

# FPS smoothing
smoothed_fps = 0.0


def extract_jpeg(stream_chunk):
    """Extract one JPEG frame from MJPEG stream."""
    global jpeg_data
    jpeg_data += stream_chunk
    start = jpeg_data.find(b"\xff\xd8")
    end = jpeg_data.find(b"\xff\xd9")
    if start != -1 and end != -1:
        frame = jpeg_data[start:end + 2]
        jpeg_data = jpeg_data[end + 2:]
        return frame
    return None


# ---------------------------------------------------
# Main loop
# ---------------------------------------------------
frame_count = 0

try:
    while True:
        chunk = proc.stdout.read(1024)
        if not chunk:
            continue

        jpeg = extract_jpeg(chunk)
        if jpeg is None:
            continue

        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue

        # Work in a smaller 320x320 space
        frame_small = cv2.resize(frame, (320, 320))
        h_small, w_small = frame_small.shape[:2]
        stop_line_y = int(STOP_LINE_Y_REL * h_small)

        frame_count += 1
        # Run YOLO every 2nd frame for speed
        if frame_count % 2 != 0:
            continue

        # ----------------- YOLO INFERENCE ----------------
        t0 = time.time()

        # Only cars: COCO class 2
        results = model(
            frame_small,
            imgsz=320,
            classes=[2],
            conf=0.4,
            verbose=False,
        )

        t1 = time.time()
        dt_inf = t1 - t0
        inst_fps = 1.0 / dt_inf if dt_inf > 0 else 0.0
        smoothed_fps = 0.8 * smoothed_fps + 0.2 * inst_fps if smoothed_fps > 0 else inst_fps

        annotated = results[0].plot()
        boxes = results[0].boxes
        now = time.time()

        # Draw stop line
        cv2.line(
            annotated,
            (0, stop_line_y),
            (w_small, stop_line_y),
            (0, 0, 255),
            2,
        )

        # Mark all tracks unmatched initially
        for tid in tracks:
            tracks[tid]["matched"] = False

        # ------------- DISTANCE-BASED TRIGGERS --------------
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()

            for box, conf in zip(xyxy, confs):
                if conf < 0.4:
                    continue

                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                # Match to existing track by nearest center
                best_id = None
                best_dist_px = MAX_MATCH_DIST
                for tid, tr in tracks.items():
                    last_cx, last_cy = tr["center"]
                    dx = cx - last_cx
                    dy = cy - last_cy
                    dist_px = (dx * dx + dy * dy) ** 0.5
                    if dist_px < best_dist_px:
                        best_dist_px = dist_px
                        best_id = tid

                if best_id is None:
                    # New track
                    track_id = next_track_id
                    tracks[track_id] = {
                        "center": (cx, cy),
                        "last_time": now,
                        "last_dist": None,  # meters
                        "hit_30": False,
                        "hit_20": False,
                        "hit_10": False,
                        "matched": True,
                    }
                    next_track_id += 1
                else:
                    # Update existing
                    track_id = best_id
                    tr = tracks[track_id]
                    tr["center"] = (cx, cy)
                    tr["last_time"] = now
                    tr["matched"] = True

                tr = tracks[track_id]

                # Distance from car to stop line, in pixels
                # Positive if car is above (upstream of) the stop line
                dist_px = stop_line_y - cy

                # Convert to meters (can be negative if car is past the line)
                dist_m = dist_px / PIXELS_PER_METER

                # Store
                prev_dist = tr["last_dist"]
                tr["last_dist"] = dist_m

                # Draw current approximate distance above car
                if dist_m > 0:
                    dist_label = f"{dist_m:.1f} m"
                else:
                    dist_label = f"{abs(dist_m):.1f} m past line"

                text_pos = (int(cx), int(cy) - 10)
                cv2.putText(
                    annotated,
                    dist_label,
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

                # If we don't have a previous distance yet, we can't detect crossings
                if prev_dist is None:
                    continue

                # Only trigger when moving TOWARDS the line (distance decreasing)
                moving_towards = dist_m < prev_dist

                if not moving_towards:
                    continue

                # Check threshold crossings: from >T to <=T
                # 30 m
                if (prev_dist > 30.0) and (dist_m <= 30.0) and (not tr["hit_30"]):
                    tr["hit_30"] = True
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(SNAP_DIR, f"30m_track{track_id}_{ts}.jpg")
                    cv2.imwrite(filename, annotated)
                    print(f"[30m] Track {track_id} crossed ~30m. Snapshot saved: {filename}")

                # 20 m
                if (prev_dist > 20.0) and (dist_m <= 20.0) and (not tr["hit_20"]):
                    tr["hit_20"] = True
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(SNAP_DIR, f"20m_track{track_id}_{ts}.jpg")
                    cv2.imwrite(filename, annotated)
                    print(f"[20m] Track {track_id} crossed ~20m. Snapshot saved: {filename}")

                # 10 m
                if (prev_dist > 10.0) and (dist_m <= 10.0) and (not tr["hit_10"]):
                    tr["hit_10"] = True
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(SNAP_DIR, f"10m_track{track_id}_{ts}.jpg")
                    cv2.imwrite(filename, annotated)
                    print(f"[10m] Track {track_id} crossed ~10m. Snapshot saved: {filename}")

        # Cleanup old unmatched tracks
        to_delete = []
        for tid, tr in tracks.items():
            if not tr.get("matched", False) and (now - tr["last_time"] > 2.0):
                to_delete.append(tid)
        for tid in to_delete:
            del tracks[tid]

        # ---------------- DRAW GLOBAL FPS ------------------
        fps_text = f"FPS: {smoothed_fps:.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(fps_text, font, 1, 2)[0]
        text_x = annotated.shape[1] - text_size[0] - 10
        text_y = text_size[1] + 10
        cv2.putText(
            annotated,
            fps_text,
            (text_x, text_y),
            font,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # ---------------- SHOW FRAME -----------------------
        cv2.imshow("RPICAM - Distance to Stop Line", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    proc.terminate()
    cv2.destroyAllWindows()

