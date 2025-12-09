import cv2
import numpy as np
from Color_Detection_Files.color_detection import detect_light_color as original_detect_light_color

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
# e.g., adaptive_hsv['red'] = [ (np.array([0,100,100]), np.array([10,255,255])),
#                              (np.array([170,100,100]), np.array([179,255,255])) ]
adaptive_hsv = {"red": None, "yellow": None, "green": None}

# OFF baseline values per color (from same calibration freeze)
off_baseline = {"red": None, "yellow": None, "green": None}

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
    # If the interval crosses the 0/179 boundary (e.g., h_low=170, h_high=10), split into two ranges.
    # Because our H is 0..179, check if h_high < h_low (wrap) OR if range is "large" artificially.
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
def detect_light_color(img, adaptive_bounds=None, show_hsv=False):
    """
    Wrapper around original detection:
      - If adaptive_bounds provided for a color, use that HSV ranges to test for presence.
      - Otherwise fall back to original detection routine (which has static color ranges).
    Returns: 'red'/'yellow'/'green'/'None'
    """
    # If there is no adaptive_bounds dict or it's empty, fall back
    if adaptive_bounds is None or not any(adaptive_bounds.values()):
        return original_detect_light_color(img, show_hsv=show_hsv)

    if img is None or img.size == 0:
        return "None"

    roi_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    detected = {}
    MIN_AREA = 3

    # For each color, if we have adaptive ranges, use them.
    for color in ("red", "yellow", "green"):
        ranges = adaptive_bounds.get(color)
        if ranges is None:
            continue

        total_area = 0
        for (low, high) in ranges:
            mask = cv2.inRange(roi_hsv, low, high)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area > MIN_AREA:
                    total_area += area

        if total_area > 0:
            detected[color] = total_area

    if detected:
        # return color with largest area
        return max(detected, key=detected.get)
    return "None"

# ---------------------------
# Main loop
# ---------------------------
def main():
    global drawing, ix, iy, fx, fy, temp_rois, roi_confirmed, selecting_target, adaptive_hsv, off_baseline

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
                return detect_light_color(roi_frame, adaptive_bounds={color_key: bounds} if bounds else None, show_hsv=False)

            red_state = draw_and_detect("red")
            yellow_state = draw_and_detect("yellow")
            green_state = draw_and_detect("green")

            cv2.putText(display, f"RED:    {red_state}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            cv2.putText(display, f"YELLOW: {yellow_state}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            cv2.putText(display, f"GREEN:  {green_state}", (10, 100),
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
                        # draw a small patch showing the median color
                        med_h = int(np.median([ (r[0][0]+r[1][0])//2 for r in ranges ]))
                        med_s = int(np.median([ (r[0][1]+r[1][1])//2 for r in ranges ]))
                        med_v = int(np.median([ (r[0][2]+r[1][2])//2 for r in ranges ]))
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

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()