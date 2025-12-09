import cv2
import subprocess
import numpy as np
import time
import os
import socket
import threading
from ultralytics import YOLO

# ---------------------------------------------------
# DISTANCE CALIBRATION (TUNE THIS!)
# ---------------------------------------------------
PIXELS_PER_METER = 50  # <-- placeholder, adjust after seeing real camera view

# Thresholds (meters) upstream of the stop line
THRESHOLDS = [3.0, 2.0, 1.0]  # [3m, 2m, 1m]

# ---------------------------------------------------
# Arduino Serial Configuration
# ---------------------------------------------------
ARDUINO_PORT = '/dev/ttyACM0'  # Change to your Arduino's port (could be /dev/ttyACM0)
ARDUINO_BAUD = 19200

# ---------------------------------------------------
# Socket Listener Configuration
# ---------------------------------------------------
LISTENER_HOST = '192.168.2.2'  # Bookworm's IP
LISTENER_PORT = 5000

# ---------------------------------------------------
# General setup
# ---------------------------------------------------
SNAP_DIR = "snapshots"
os.makedirs(SNAP_DIR, exist_ok=True)

# Global state for traffic light control
current_light = "RED"  # Start with RED (active for detection)
light_lock = threading.Lock()

# Initialize Arduino serial connection
try:
    arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=1)
    time.sleep(2)  # Wait for Arduino to reset
    print(f"[SERIAL] Connected to Arduino on {ARDUINO_PORT}")
except Exception as e:
    print(f"[SERIAL] Failed to connect to Arduino: {e}")
    arduino = None

def send_alert_to_arduino():
    """Send ALERT command to Arduino"""
    if arduino is not None:
        try:
            arduino.write(b"ALERT\n")
            arduino.flush()
            print("[SERIAL] Sent ALERT to Arduino")
        except Exception as e:
            print(f"[SERIAL] Error sending to Arduino: {e}")
    else:
        print("[SERIAL] Arduino not connected - cannot send ALERT")

def listener_thread():
    """Background thread to listen for RED/GREEN commands"""
    global current_light
    
    print("Listener thread started. Waiting for commands...")
    
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((LISTENER_HOST, LISTENER_PORT))
                s.listen()
                
                conn, addr = s.accept()
                with conn:
                    data = conn.recv(1024).decode().strip().upper()
                    print(f"\n[LISTENER] Received command: {data}")
                    
                    with light_lock:
                        if data == "YES":  # RED light
                            current_light = "RED"
                            print("[LISTENER] Traffic light is RED - Detection ACTIVE")
                        elif data == "NO":  # GREEN light
                            current_light = "GREEN"
                            print("[LISTENER] Traffic light is GREEN - Detection INACTIVE")
                        else:
                            print(f"[LISTENER] Unknown command: {data}")
        except Exception as e:
            print(f"[LISTENER] Error: {e}")
            time.sleep(1)  # Wait before retrying

# Start listener thread
listener = threading.Thread(target=listener_thread, daemon=True)
listener.start()

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
tracks = {}
next_track_id = 0
MAX_MATCH_DIST = 50  # pixels

# Stop line location in relative image coordinates
STOP_LINE_X_REL = 0.7  # 70% across the image (vertical line)

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

        # Work in a 640x640 space
        frame_small = cv2.resize(frame, (320, 320))
        h_small, w_small = frame_small.shape[:2]
        stop_line_x = int(STOP_LINE_X_REL * w_small)

        frame_count += 1
        
        # Check current light status
        with light_lock:
            is_red = (current_light == "RED")
        
        # Run YOLO every 2nd frame for speed (only when RED)
        if frame_count % 2 != 0:
            continue

        # ----------------- YOLO INFERENCE ----------------
        t0 = time.time()

        # Only run detection if light is RED
        if is_red:
            results = model(
                frame_small,
                imgsz=320,
                classes=[2],  # Only cars: COCO class 2
                conf=0.4,
                verbose=False,
            )
            annotated = results[0].plot()
            boxes = results[0].boxes
        else:
            # Create empty results when GREEN
            annotated = frame_small.copy()
            boxes = None

        t1 = time.time()
        dt_inf = t1 - t0
        inst_fps = 1.0 / dt_inf if dt_inf > 0 else 0.0
        smoothed_fps = 0.8 * smoothed_fps + 0.2 * inst_fps if smoothed_fps > 0 else inst_fps

        now = time.time()

        # Draw stop line
        cv2.line(
            annotated,
            (stop_line_x, 0),
            (stop_line_x, h_small),
            (0, 0, 255),
            2,
        )

        # Display current light status
        light_color = (0, 0, 255) if is_red else (0, 255, 0)  # Red or Green
        light_text = f"Light: {current_light}"
        cv2.putText(
            annotated,
            light_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            light_color,
            2,
            cv2.LINE_AA,
        )

        # Mark all tracks unmatched initially
        for tid in tracks:
            tracks[tid]["matched"] = False

        # RISK LEVEL FOR THIS FRAME (0 = none, 1=3m, 2=2m, 3=1m/past)
        risk_level = 0

        # ------------- DISTANCE-BASED TRIGGERS (only when RED) --------------
        if is_red and boxes is not None and len(boxes) > 0:
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
                        "hit_3": False,
                        "hit_2": False,
                        "hit_1": False,
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
                dist_px = stop_line_x - cx

                # Convert to meters
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

                # ---- Compute RISK MODE based on current distance ----
                # Treat past-the-line (dist_m <= 0) as highest risk (1m mode)
                if dist_m <= 0 or dist_m <= THRESHOLDS[2]:
                    risk_level = max(risk_level, 3)  # 1m / past line
                elif dist_m <= THRESHOLDS[1]:
                    risk_level = max(risk_level, 2)  # 2m
                elif dist_m <= THRESHOLDS[0]:
                    risk_level = max(risk_level, 1)  # 3m

                # If we don't have a previous distance yet, we can't detect crossings
                if prev_dist is None:
                    continue

                # Only trigger when moving TOWARDS the line (distance decreasing)
                moving_towards = dist_m < prev_dist

                if not moving_towards:
                    continue

                # Check threshold crossings: from >T to <=T

                # 3 m
                if (prev_dist > 3.0) and (dist_m <= 3.0) and (not tr["hit_3"]):
                    tr["hit_3"] = True
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(SNAP_DIR, f"3m_track{track_id}_{ts}.jpg")
                    cv2.imwrite(filename, annotated)
                    print(f"\n*** ALERT DETECTED *** [3m] Track {track_id} crossed ~3m. Snapshot: {filename}")

                # 2 m
                if (prev_dist > 2.0) and (dist_m <= 2.0) and (not tr["hit_2"]):
                    tr["hit_2"] = True
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(SNAP_DIR, f"2m_track{track_id}_{ts}.jpg")
                    cv2.imwrite(filename, annotated)
                    print(f"\n*** ALERT DETECTED *** [2m] Track {track_id} crossed ~2m. Snapshot: {filename}")
                    send_alert_to_arduino()  # Send to Arduino
                    
                # 1 m
                if (prev_dist > 1.0) and (dist_m <= 1.0) and (not tr["hit_1"]):
                    tr["hit_1"] = True
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(SNAP_DIR, f"1m_track{track_id}_{ts}.jpg")
                    cv2.imwrite(filename, annotated)
                    print(f"\n*** ALERT DETECTED *** [1m] Track {track_id} crossed ~1m. Snapshot: {filename}")
                    send_alert_to_arduino()  # Send to Arduino
           
        elif not is_red:
            # When GREEN, display "No Alert"
            cv2.putText(
                annotated,
                "No Alert - GREEN Light",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # ---------------- RISK MESSAGE (for RED light) ----------------
        if is_red:
            if risk_level == 0:
                risk_msg = "RISK: No vehicle within 3 m"
                risk_color = (0, 255, 0)  # green
            elif risk_level == 1:
                risk_msg = "RISK MODE: 3 m - Vehicle approaching"
                risk_color = (0, 255, 255)  # yellow
            elif risk_level == 2:
                risk_msg = "RISK MODE: 2 m - High risk"
                risk_color = (0, 165, 255)  # orange-ish
            else:  # risk_level == 3
                risk_msg = "RISK MODE: 1 m - IMMINENT RISK"
                risk_color = (0, 0, 255)  # red

            cv2.putText(
                annotated,
                risk_msg,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                risk_color,
                2,
                cv2.LINE_AA,
            )

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

