import cv2
import numpy as np
import socket
import time
import sys

# --- Network Configuration ---
BOOKWORM_HOST = '192.168.2.2'  # Bookworm Pi IP
BOOKWORM_PORT = 5000

def send_command(command):
    """Send YES or NO command to Bookworm Pi"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((BOOKWORM_HOST, BOOKWORM_PORT))
            s.sendall(command.encode())
            print(f"[{time.strftime('%H:%M:%S')}] ✅ Sent: {command}")
            return True
    except Exception as e:
        print(f"❌ Error sending command: {e}")
        return False

# ---------------------------
# Brightness normalization
# ---------------------------
def normalize_brightness(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# ---------------------------
# Globals for drawing
# ---------------------------
drawing = False
ix = iy = fx = fy = -1

# Temporary drawn ROIs waiting for ENTER (or ESC)
temp_rois = {"red": None, "yellow": None, "green": None}

# Confirmed/saved ROIs used for detection
roi_confirmed = {"red": None, "yellow": None, "green": None}

# Currently active selection target while frozen (for ROI drawing)
selecting_target = None

# Adaptive HSV bounds from calibration: maps color -> list of (lower, upper) ranges
adaptive_hsv = {"red": None, "yellow": None, "green": None}

# OFF baseline values per color (from same calibration freeze)
off_baseline = {"red": None, "yellow": None, "green": None}

# Track last sent command to avoid spamming
last_sent_command = None

# mouse callback for ROI drawing
def mouse_draw(event, x, y, flags, param):
    """Mouse callback: update current drawing coordinates and set temp_rois for the active target."""
    global drawing, ix, iy, fx, fy, temp_rois, selecting_target

    if selecting_target is None:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        fx, fy = x, y
        temp_rois[selecting_target] = None

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        fx, fy = x, y

    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False
        fx, fy = x, y
        x1, y1 = min(ix, fx), min(iy, fy)
        x2, y2 = max(ix, fx), max(iy, fy)
        # guard minimal size
        w, h = max(2, x2 - x1), max(2, y2 - y1)
        temp_rois[selecting_target] = (x1, y1, w, h)

# ---------------------------
# Adaptive HSV computation (Option C single-freeze extraction)
# ---------------------------
def compute_adaptive_hsv_from_roi(roi_bgr):
    """
    Given a BGR ROI (single freeze, user indicated this ROI contains an ON light),
    compute an adaptive HSV range (one or two ranges for hue if wraparound).
    Returns: list of (lower_np_array, upper_np_array), off_baseline_v
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return None, None

    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h = roi_hsv[:, :, 0].flatten().astype(np.int32)  # 0..179
    s = roi_hsv[:, :, 1].flatten().astype(np.int32)
    v = roi_hsv[:, :, 2].flatten().astype(np.int32)

    # Basic cleaning: ignore extremely dark pixels for ON candidate selection
    # We'll pick ON pixels as those with high V and moderate S
    v70 = np.percentile(v, 70)
    s30 = np.percentile(s, 30)
    on_mask = (v >= v70) & (s >= s30)

    # Fallback if too few pixels in on_mask
    if np.count_nonzero(on_mask) < max(10, 0.01 * v.size):
        # choose top 30% by V
        vth = np.percentile(v, 70)
        on_mask = v >= vth

    on_h = h[on_mask]
    on_s = s[on_mask]
    on_v = v[on_mask]

    if on_h.size == 0:
        return None, None

    # percentiles for ON color bounds (5th and 95th)
    h_low_p = np.percentile(on_h, 5)
    h_high_p = np.percentile(on_h, 95)
    s_low_p = np.percentile(on_s, 5)
    s_high_p = np.percentile(on_s, 95)
    v_low_p = np.percentile(on_v, 5)
    v_high_p = np.percentile(on_v, 95)

    # Build HSV range(s). Handle hue wraparound:
    ranges = []
    h_low = int(round(h_low_p))
    h_high = int(round(h_high_p))
    s_low = int(round(s_low_p))
    s_high = int(round(s_high_p))
    v_low = int(round(v_low_p))
    v_high = int(round(v_high_p))

    # Clamp s and v to valid range
    s_low = max(0, min(255, s_low))
    s_high = max(0, min(255, s_high))
    v_low = max(0, min(255, v_low))
    v_high = max(0, min(255, v_high))

    if h_high >= h_low:
        ranges.append((np.array([h_low, s_low, v_low], dtype=np.uint8),
                       np.array([h_high, s_high, v_high], dtype=np.uint8)))
    else:
        # wrap case: two ranges: [0, h_high] and [h_low, 179]
        ranges.append((np.array([0, s_low, v_low], dtype=np.uint8),
                       np.array([h_high, s_high, v_high], dtype=np.uint8)))
        ranges.append((np.array([h_low, s_low, v_low], dtype=np.uint8),
                       np.array([179, s_high, v_high], dtype=np.uint8)))

    # Also compute a simple OFF baseline (median V of darker pixels)
    v30 = np.percentile(v, 30)
    off_mask = v <= v30
    if np.count_nonzero(off_mask) == 0:
        off_baseline_v = float(np.percentile(v, 10))
    else:
        off_baseline_v = float(np.median(v[off_mask]))

    return ranges, off_baseline_v

# ---------------------------
# Modified detect function wrapper that uses adaptive HSV bounds when present
# ---------------------------
def detect_light_color(
        roi_frame,
        adaptive_bounds=None,
        show_hsv=False,
        USE_HARD_CODED_BACKUP=True):
    """
    Drop-in replacement light color detector.

    Parameters:
        roi_frame: the ROI portion of the frame (already cropped).
        adaptive_bounds: dict like {"red": [(h_low, h_high, s_low, ...), ...]}
                         created during calibration.
        show_hsv: if True, prints mean HSV values for debugging.
        USE_HARD_CODED_BACKUP: if True, falls back to fixed HSV ranges.

    Returns:
        "red", "yellow", "green", or "none"
    """

    # -----------------------------------------
    # 1. Convert the ROI to HSV
    # -----------------------------------------
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

    if show_hsv:
        h_mean = int(hsv[:, :, 0].mean())
        s_mean = int(hsv[:, :, 1].mean())
        v_mean = int(hsv[:, :, 2].mean())
        print(f"HSV Debug → H:{h_mean}  S:{s_mean}  V:{v_mean}")

    # -----------------------------------------
    # 2. ADAPTIVE HSV DETECTION
    # -----------------------------------------
    if adaptive_bounds is None:
        adaptive_bounds = {}

    adaptive_scores = {}

    for color_key, ranges in adaptive_bounds.items():
        mask_total = None

        for (lower, upper) in ranges:
            mask = cv2.inRange(hsv, lower, upper)

            if mask_total is None:
                mask_total = mask
            else:
                mask_total = cv2.bitwise_or(mask_total, mask)

        adaptive_scores[color_key] = cv2.countNonZero(mask_total)

    # Choose max-scoring adaptive color
    if adaptive_scores:
        best_adaptive = max(adaptive_scores, key=adaptive_scores.get)
        best_adaptive_score = adaptive_scores[best_adaptive]
    else:
        best_adaptive = None
        best_adaptive_score = 0

    # Threshold for confidence
    if best_adaptive_score > 50:
        return best_adaptive

    # -----------------------------------------
    # 3. HARD-CODED HSV BACKUP
    # -----------------------------------------
    if not USE_HARD_CODED_BACKUP:
        return "none"  # No fallback → adaptive mode only

    hardcoded_ranges = {
        "red": [
            ((0, 120, 80), (10, 255, 255)),
            ((170, 120, 80), (180, 255, 255))
        ],
        "yellow": [
            ((20, 120, 80), (35, 255, 255))
        ],
        "green": [
            ((40, 80, 80), (85, 255, 255))
        ]
    }

    backup_scores = {}

    for color, ranges in hardcoded_ranges.items():
        mask_total = None

        for (low, high) in ranges:
            mask = cv2.inRange(hsv, low, high)
            if mask_total is None:
                mask_total = mask
            else:
                mask_total = cv2.bitwise_or(mask_total, mask)

        backup_scores[color] = cv2.countNonZero(mask_total)

    best_backup = max(backup_scores, key=backup_scores.get)
    best_backup_score = backup_scores[best_backup]

    if best_backup_score > 50:
        return best_backup

    return "none"

# ---------------------------
# Main loop
# ---------------------------
def main():
    global drawing, ix, iy, fx, fy, temp_rois, roi_confirmed, selecting_target, adaptive_hsv, off_baseline, last_sent_command

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    cv2.namedWindow("Traffic Light Detection")
    cv2.setMouseCallback("Traffic Light Detection", mouse_draw)

    selecting = False         # for ROI drawing freeze
    freeze_frame = None
    mode = "live"             # "live", "roi_select", "calibrate"
    calibrate_active = False  # when user pressed 'c' and is in calibration freeze

    print("Controls:")
    print("  R/Y/G - enter ROI selection for Red/Yellow/Green (screen freezes).")
    print("  Drag with left mouse to draw a rectangle.")
    print("  Press ENTER to confirm ALL temporary ROIs.")
    print("  Press ESC to cancel the CURRENT target's temp selection.")
    print("  Press C to enter calibration freeze (single freeze; press R/Y/G to mark which lights are ON).")
    print("  While calibrating you may press R/Y/G multiple times to compute bounds for those colors,")
    print("    then press ENTER to finish calibration and return to live detection.")
    print("  Press Q to quit.\n")

    try:
        while True:
            if mode == "live" and not selecting:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = normalize_brightness(frame)
                display = frame.copy()
            else:
                # when frozen (either ROI select or calibration), show freeze_frame copy
                display = freeze_frame.copy()
                txt = "Selecting ROIs (ENTER confirm) " if mode == "roi_select" else "Calibration mode (press R/Y/G to mark ON lights, ENTER to finish)"
                cv2.putText(display, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

                if drawing:
                    cv2.rectangle(display, (ix, iy), (fx, fy), (0,255,255), 2)

                for color_key, rect in temp_rois.items():
                    if rect is not None:
                        x,y,w,h = rect
                        col = (0,0,255) if color_key=="red" else (0,255,255) if color_key=="yellow" else (0,255,0)
                        cv2.rectangle(display, (x,y), (x+w, y+h), col, 2)
                        cv2.putText(display, f"TEMP {color_key.upper()}", (x, y-6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

                for color_key, rect in roi_confirmed.items():
                    if rect is None:
                        continue
                    # don't draw as confirmed when actively selecting that target (so user sees temp)
                    if selecting_target == color_key and mode == "roi_select":
                        continue
                    x,y,w,h = rect
                    col = (0,0,255) if color_key=="red" else (0,255,255) if color_key=="yellow" else (0,255,0)
                    cv2.rectangle(display, (x,y), (x+w, y+h), col, 1)
                    cv2.putText(display, f"CONF {color_key.upper()}", (x, y-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

            # When live, draw detection boxes and run detection
            if mode == "live" and not selecting:
                def draw_and_detect(color_key):
                    rect = roi_confirmed[color_key]
                    if rect is None:
                        return "None"
                    x,y,w,h = rect
                    col = (0,0,255) if color_key=="red" else (0,255,255) if color_key=="yellow" else (0,255,0)
                    cv2.rectangle(display, (x,y), (x+w, y+h), col, 2)
                    roi_frame = frame[y:y+h, x:x+w]

                    # pass adaptive bounds if available
                    bounds = adaptive_hsv.get(color_key)
                    detected_color = detect_light_color(
                        roi_frame,
                        adaptive_bounds={color_key: bounds} if bounds else None,
                        show_hsv=False,
                        USE_HARD_CODED_BACKUP=False
                    )
                    
                    return color_key if detected_color == color_key else "off"

                red_state = draw_and_detect("red")
                yellow_state = draw_and_detect("yellow")
                green_state = draw_and_detect("green")
                
                # --- Network Command Logic ---
                # Send YES when RED or YELLOW is detected
                # Send NO when GREEN is detected
                command_to_send = None
                
                if red_state == "red" or yellow_state == "yellow":
                    command_to_send = "yes"
                elif green_state == "green":
                    command_to_send = "no"
                
                # Only send if command changed (avoid spamming)
                if command_to_send and command_to_send != last_sent_command:
                    if send_command(command_to_send):
                        last_sent_command = command_to_send
                        # Visual feedback
                        feedback_text = "Command: YES (RED/YELLOW)" if command_to_send == "yes" else "Command: NO (GREEN)"
                        feedback_color = (0, 0, 255) if command_to_send == "yes" else (0, 255, 0)
                        cv2.putText(display, feedback_text, (10, 135),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, feedback_color, 2)

                cv2.putText(display, f"RED:    {red_state.upper()}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                cv2.putText(display, f"YELLOW: {yellow_state.upper()}", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                cv2.putText(display, f"GREEN:  {green_state.upper()}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            cv2.imshow("Traffic Light Detection", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            # ROI selection keys
            if key in (ord('r'), ord('y'), ord('g')):
                # Determine which action mode we are in:
                if mode == "live":
                    # Enter ROI selection mode
                    new_target = ("red" if key == ord('r') else "yellow" if key == ord('y') else "green")
                    # clear the confirmed roi for this color (so the user can redraw)
                    roi_confirmed[new_target] = None
                    selecting_target = new_target
                    mode = "roi_select"
                    selecting = True
                    drawing = False
                    ix = iy = fx = fy = -1
                    temp_rois[selecting_target] = None
                    # freeze raw camera frame (not display)
                    ret, frame = cap.read()
                    frame = normalize_brightness(frame) if ret else frame
                    freeze_frame = frame.copy()
                    print(f"Entered ROI selection mode for {selecting_target.upper()}.")
                elif mode == "roi_select":
                    # If already selecting ROIs and user presses another color key, start selecting that target
                    new_target = ("red" if key == ord('r') else "yellow" if key == ord('y') else "green")
                    if selecting and new_target == selecting_target:
                        print(f"Ignoring repeated region key '{new_target.upper()}' during selection.")
                        continue
                    selecting_target = new_target
                    drawing = False
                    ix = iy = fx = fy = -1
                    temp_rois[selecting_target] = None
                    print(f"Switched ROI selection target to {selecting_target.upper()}.")
                elif mode == "calibrate":
                    # In calibration freeze, pressing R/Y/G instructs the code to compute adaptive HSV for that color
                    color_key = ("red" if key == ord('r') else "yellow" if key == ord('y') else "green")
                    rect = roi_confirmed[color_key]
                    if rect is None:
                        print(f"No confirmed ROI for {color_key.upper()} — draw and confirm ROI first.")
                    else:
                        x,y,w,h = rect
                        roi_frame = freeze_frame[y:y+h, x:x+w].copy()
                        ranges, off_v = compute_adaptive_hsv_from_roi(roi_frame)
                        if ranges is None:
                            print(f"Calibration failed for {color_key.upper()}: couldn't compute ranges from ROI.")
                        else:
                            adaptive_hsv[color_key] = ranges
                            off_baseline[color_key] = off_v
                            print(f"Calibrated {color_key.upper()} — {len(ranges)} HSV range(s), off_V={off_v:.1f}")
                            # Draw the ranges on the freeze_frame for feedback:
                            med_h = int(np.median([ (int(r[0][0]) + int(r[1][0])) // 2 for r in ranges ]))
                            med_s = int(np.median([ (int(r[0][1]) + int(r[1][1])) // 2 for r in ranges ]))
                            med_v = int(np.median([ (int(r[0][2]) + int(r[1][2])) // 2 for r in ranges ]))
                            patch = np.full((30, 60, 3), (0,0,0), dtype=np.uint8)
                            patch_hsv = np.uint8([[[med_h, med_s, med_v]]])
                            patch_bgr = cv2.cvtColor(patch_hsv, cv2.COLOR_HSV2BGR)[0,0].tolist()
                            cv2.rectangle(freeze_frame, (x, max(0,y-40)), (x+60, max(0,y-10)), patch_bgr, -1)

            elif key == ord('c'):   # calibration freeze
                if mode == "live":
                    # freeze raw frame for calibration
                    ret, frame = cap.read()
                    frame = normalize_brightness(frame) if ret else frame
                    freeze_frame = frame.copy()
                    mode = "calibrate"
                    selecting = False
                    selecting_target = None
                    drawing = False
                    print("Entered CALIBRATION freeze. Press R/Y/G to indicate which lights are ON. Press ENTER when done.")
                else:
                    print("Already frozen — press ENTER to finish current mode first.")

            elif key == 13:  # ENTER
                if mode == "roi_select":
                    committed = []
                    for color_key, rect in temp_rois.items():
                        if rect is not None:
                            roi_confirmed[color_key] = rect
                            committed.append(color_key)
                            temp_rois[color_key] = None
                    # exit ROI selection
                    mode = "live"
                    selecting = False
                    selecting_target = None
                    drawing = False
                    if committed:
                        print("Confirmed ROIs for:", ", ".join(c.upper() for c in committed))
                    else:
                        print("ENTER pressed but no temporary ROI to confirm.")
                elif mode == "calibrate":
                    # finish calibration and return to live
                    mode = "live"
                    selecting = False
                    selecting_target = None
                    drawing = False
                    print("Calibration complete — returning to live detection.")
                else:
                    # nothing to confirm in live
                    pass

            elif key == 27:  # ESC
                if mode == "roi_select":
                    if selecting_target is not None:
                        print(f"Canceled selection for {selecting_target.upper()}")
                        temp_rois[selecting_target] = None
                    mode = "live"
                    selecting = False
                    selecting_target = None
                    drawing = False
                elif mode == "calibrate":
                    print("Canceled calibration and returning to live.")
                    mode = "live"
                    selecting = False
                    selecting_target = None
                    drawing = False

    except KeyboardInterrupt:
        print("\nExiting script...")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
