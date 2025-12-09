#CURRENT COLOR DETECTION CODE

import cv2
import numpy as np

def detect_light_color(img, show_hsv=False):
    """
    Detects whether red, yellow, or green light is ON in the given cropped traffic light image.
    Optionally prints average HSV values for debugging.
    Returns the color name as a string.
    """
    if img is None or img.size == 0:
        return "None"
    
    # Convert to HSV
    roi_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Compute average HSV for debugging
    if show_hsv:
        avg_h = np.mean(roi_hsv[:, :, 0])
        avg_s = np.mean(roi_hsv[:, :, 1])
        avg_v = np.mean(roi_hsv[:, :, 2])
        print(f"[DEBUG] Avg HSV -> H: {avg_h:.1f}, S: {avg_s:.1f}, V: {avg_v:.1f}")

    # Detection parameters
    #MIN_BRIGHTNESS = 0
    #MIN_SATURATION = 0
    MIN_AREA = 3

    color_ranges = {
        'red': [
            ([0,100,100], [10, 255, 255]),
            ([170, 100 , 100], [180, 255, 255])
        ],
        'yellow': [
            #([20, 120, 160], [35, 255, 255]),
            ([20, 60, 160], [70, 255, 255])
        ],
        'green': [
            #([45, 100, 100], [75, 255, 255]),
            ([75, 10, 100], [100, 255, 255])
        ]
    }

    detected_colors = {}

    for color, ranges in color_ranges.items():
        total_area = 0
        for lower, upper in ranges:
            lower_bound = np.array(lower, dtype=np.uint8)
            upper_bound = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(roi_hsv, lower_bound, upper_bound)

            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > MIN_AREA:
                    total_area += area

        if total_area > 0:
            detected_colors[color] = total_area

    if detected_colors:
        detected_color = max(detected_colors, key=detected_colors.get)
        if show_hsv:
            print(f"[DEBUG] Detected color: {detected_color.upper()} (area = {detected_colors[detected_color]:.1f})")
        return detected_color

    return "None"
message.txt
3 KB