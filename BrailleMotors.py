# ============================================================
# Geometric Spline-Based 6-Motor Dice Pattern Selector
# WITH CENTROID FITTING, STABLE FOOT DETECTION, AND FOOT LIFT
# WITH MOTOR OUTPUT FUNCTIONALITY
# ============================================================
# Version: 2.2 (Added Motor Output)
# 
# SYSTEM OVERVIEW:
# ================
# 
# Core Algorithm Stages:
# ----------------------
# 1. FOOT DETECTION
#    - Read pressure data from sensor grid
#    - Apply gaussian filter and thresholding
#    - Extract foot contour using edge detection
# 
# 2. BISECTOR COMPUTATION
#    - Calculate medial axis of foot
#    - Determine heel-to-toe direction
#    - Temporal smoothing for stability
# 
# 3. EQUAL-AREA SPLINE
#    - Compute smooth geometric spline
#    - Splits foot into equal left/right areas
#    - Heavy smoothing to avoid wiggly curves
# 
# 4. BRAILLE MOTOR SELECTION (6 motors)
#    - Find motor pairs perpendicular to spline
#    - Test multiple rectangle orientations (-60° to +60°)
#    - Select 3 best pairs (heel/mid/toe pattern)
#    - Enforce uniqueness and minimum separation
#    - Map motors to braille dots (1-6) based on foot orientation
# 
# 5. CENTROID MOTOR SELECTION (12 motors)
#    - Calculate foot centroid (center of mass)
#    - Find 12 nearest motors to centroid
#    - Displayed as small red dots
# 
# 6. STABILITY DETECTION (LENIENT)
#    - Track braille motor history over 1 second
#    - Use Jaccard similarity to compare configurations
#    - Trigger when average similarity ≥ 85% (roughly 5/6 motors)
#    - Allows minor fluctuations, catches true instability
# 
# 7. DANGER DISPLAY SEQUENCE
#    - Show both motor sets for 1 second
#    - Display "DANGER" in Grade 1 Braille letter-by-letter
#    - All 6 braille motors visible throughout
#    - Active motors flash lime → white (2 sec per letter)
#    - 0.3s pause between letters
# 
# 8. FROZEN STATE & FOOT LIFT
#    - After sequence, enter FROZEN state
#    - Motors stay displayed, no new detections
#    - Monitor for foot lift (pressure/contour checks)
#    - Exit to NORMAL when foot removed
#
# 9. MOTOR OUTPUT (NEW)
#    - Prints selected centroid motors (12) during normal operation
#    - Prints selected braille motors (6) during normal operation
#    - Prints comprehensive summary when stability detected
#    - Prints active motors for each letter in DANGER sequence
# 
# ============================================================

import serial, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_fill_holes
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.integrate import quad
from skimage import measure, morphology
from matplotlib.path import Path
from collections import deque

# ---------- USER CONFIG ----------
PORT = 'COM3'
BAUD = 115200
GRID_SIZE = 16
GRID_SIZE_CM = 28.0
CM_PER_PIXEL = GRID_SIZE_CM / GRID_SIZE
REFRESH_DELAY = 0.05
SIGMA = 1.8
BASELINE_FRAMES = 40

# Motor grid configuration
MOTOR_ROWS = 5
MOTOR_COLS = 6
MOTOR_Y_SPACING = 4.66
MOTOR_X_SPACING = 4.0

grid_width = (MOTOR_COLS - 1) * MOTOR_X_SPACING
grid_height = (MOTOR_ROWS - 1) * MOTOR_Y_SPACING
x_offset = (GRID_SIZE_CM - grid_width) / 2
y_offset = (GRID_SIZE_CM - grid_height) / 2

MOTOR_GRID = [(x * MOTOR_X_SPACING + x_offset, y * MOTOR_Y_SPACING + y_offset) 
              for y in range(MOTOR_ROWS) 
              for x in range(MOTOR_COLS)]
MOTOR_COORDS = np.array(MOTOR_GRID)

# ---------- BRAILLE CONFIGURATION ----------
# Grade 1 Braille patterns (dots 1-6)
# Layout: 1 4
#         2 5
#         3 6
BRAILLE_PATTERNS = {
    'D': [1, 4, 5],
    'A': [1],
    'N': [1, 3, 4, 5],
    'G': [1, 2, 4, 5],
    'E': [1, 5],
    'R': [1, 2, 3, 5]
}

# ---------- STABILITY AND TIMING CONFIGURATION ----------
centroid_motor_history = deque(maxlen=100)
braille_motor_history = deque(maxlen=100)

STABILITY_DURATION = 2.0
BRAILLE_SIMILARITY_THRESHOLD = 0.85
LETTER_DISPLAY_DURATION = 2.0
INTER_LETTER_WAIT = 0.3

# ---------- FOOT LIFT DETECTION CONFIGURATION ----------
FOOT_LIFT_CHECK_INTERVAL = 0.1
MIN_CONTOUR_AREA = 50
MIN_PRESSURE_THRESHOLD = 20

# ---------- SERIAL ----------
print(f"Opening serial port {PORT} ...")
ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)
print("Connected successfully!\n")

# ---------- PLOT ----------
plt.ion()
fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
ax.set_facecolor('black')

# ---------- BASELINE CAPTURE ----------
print("Capturing baseline (stand clear)...")
baseline = []
for _ in range(BASELINE_FRAMES):
    ser.write(b'A')
    data = ser.read(GRID_SIZE * GRID_SIZE)
    if len(data) == GRID_SIZE * GRID_SIZE:
        frame = np.frombuffer(data, dtype=np.uint8).reshape((GRID_SIZE, GRID_SIZE))
        baseline.append(frame)
baseline_mean = np.mean(baseline, axis=0)
print("Baseline captured.\n")

# Temporal smoothing state
prev_axis = None
prev_heel = None
prev_toe = None
prev_motor_indices = None
prev_config_score = None
SMOOTHING_ALPHA = 0.6
MOTOR_SWITCH_THRESHOLD = 0.75

# System state
system_state = "NORMAL"
frozen_braille_motors = None
frozen_centroid_motors = None
frozen_braille_map = None
frozen_contour = None
frozen_heel = None
frozen_toe = None

# ============================================================
# --- FUNCTIONS ---
# ============================================================

def print_motor_output(motor_indices, label="Motors"):
    """
    Print motor indices in a formatted way.
    Converts numpy types to regular Python ints for clean output.
    
    Args:
        motor_indices: list/set of motor indices
        label: descriptive label for the output
    """
    if not motor_indices:
        print(f"{label}: None")
        return
    
    # Convert to regular Python ints (handles numpy types)
    sorted_motors = sorted([int(m) for m in motor_indices])
    print(f"{label}: {sorted_motors}")


def read_frame():
    ser.write(b'A')
    data = ser.read(GRID_SIZE * GRID_SIZE)
    if len(data) != GRID_SIZE * GRID_SIZE:
        return None
    return np.frombuffer(data, dtype=np.uint8).reshape((GRID_SIZE, GRID_SIZE))


def verify_and_correct_orientation(heel, toe, axis, contour_cm):
    """
    Verify that heel and toe are correctly identified (not swapped).
    Corrects orientation to ensure braille is never upside down.
    """
    from matplotlib.path import Path
    path = Path(contour_cm)
    
    perp = np.array([-axis[1], axis[0]])
    perp = perp / np.linalg.norm(perp)
    
    def measure_foot_width(center_point, perp_direction, search_distance=15):
        left_edge = None
        right_edge = None
        
        for dist in np.linspace(0, search_distance, 100):
            test_left = center_point - perp_direction * dist
            test_right = center_point + perp_direction * dist
            
            if left_edge is None and not path.contains_point(test_left) and dist > 0.1:
                left_edge = center_point - perp_direction * (dist - search_distance/100)
            if right_edge is None and not path.contains_point(test_right) and dist > 0.1:
                right_edge = center_point + perp_direction * (dist - search_distance/100)
            
            if left_edge is not None and right_edge is not None:
                break
        
        if left_edge is not None and right_edge is not None:
            width = np.linalg.norm(right_edge - left_edge)
            return width
        else:
            return 999.0
    
    heel_positions = [heel, heel + axis * 1.0, heel + axis * 2.0]
    toe_positions = [toe, toe - axis * 1.0, toe - axis * 2.0]
    
    heel_widths = [measure_foot_width(pos, perp) for pos in heel_positions]
    toe_widths = [measure_foot_width(pos, perp) for pos in toe_positions]
    
    avg_heel_width = np.mean([w for w in heel_widths if w < 20])
    avg_toe_width = np.mean([w for w in toe_widths if w < 20])
    
    width_difference = avg_heel_width - avg_toe_width
    SWAP_THRESHOLD = 1.0
    
    if width_difference < -SWAP_THRESHOLD:
        corrected_heel = toe
        corrected_toe = heel
        corrected_axis = -axis
        return corrected_heel, corrected_toe, corrected_axis
    else:
        return heel, toe, axis


def map_motors_to_braille_dots(motor_indices, motor_coords, axis):
    """
    Map 6 braille motors to braille dot positions (1-6) based on foot orientation.
    """
    if len(motor_indices) != 6:
        return None
    
    motors = motor_coords[motor_indices]
    center = np.mean(motors, axis=0)
    
    perp = np.array([-axis[1], axis[0]])
    
    left_motors = []
    right_motors = []
    
    for i, idx in enumerate(motor_indices):
        motor = motors[i]
        side = np.dot(motor - center, perp)
        if side < 0:
            left_motors.append((idx, motor))
        else:
            right_motors.append((idx, motor))
    
    if len(left_motors) < 3 or len(right_motors) < 3:
        sides = [np.dot(motors[i] - center, perp) for i in range(6)]
        sorted_by_side = sorted(enumerate(motor_indices), key=lambda x: sides[x[0]])
        left_motors = [(sorted_by_side[i][1], motors[sorted_by_side[i][0]]) for i in range(3)]
        right_motors = [(sorted_by_side[i][1], motors[sorted_by_side[i][0]]) for i in range(3, 6)]
    
    left_motors.sort(key=lambda x: np.dot(x[1] - center, axis))
    right_motors.sort(key=lambda x: np.dot(x[1] - center, axis))
    
    braille_map = {
        1: left_motors[0][0],
        2: left_motors[1][0],
        3: left_motors[2][0],
        4: right_motors[0][0],
        5: right_motors[1][0],
        6: right_motors[2][0]
    }
    
    return braille_map


def compute_foot_centroid(contour_cm):
    """Compute centroid of foot contour using center of mass."""
    centroid = np.mean(contour_cm, axis=0)
    return centroid


def find_nearest_motors(center_point, motor_coords, n=12):
    """Find n nearest motors to a center point."""
    distances = np.linalg.norm(motor_coords - center_point, axis=1)
    nearest_indices = np.argsort(distances)[:n]
    return list(nearest_indices)


def detect_foot_lift(blurred, contour_cm=None):
    """Detect if the foot has been lifted from the sensor."""
    max_pressure = np.max(blurred)
    if max_pressure < MIN_PRESSURE_THRESHOLD:
        return True
    
    if contour_cm is not None:
        contour_area = len(contour_cm)
        if contour_area < MIN_CONTOUR_AREA:
            return True
    
    thresh = np.median(blurred) + 0.55 * (np.max(blurred) - np.median(blurred))
    mask = blurred > thresh
    active_pixels = np.sum(mask)
    
    if active_pixels < 30:
        return True
    
    return False


def get_braille_motors_for_letter(letter, braille_map):
    """Get motor indices for displaying a letter in braille."""
    if letter not in BRAILLE_PATTERNS or braille_map is None:
        return []
    
    dots = BRAILLE_PATTERNS[letter]
    motor_indices = [braille_map[dot] for dot in dots if dot in braille_map]
    return motor_indices


def check_stability(centroid_motors, braille_motors):
    """Check if motor selections have been stable for STABILITY_DURATION seconds."""
    current_time = time.time()
    
    centroid_set = frozenset(centroid_motors)
    braille_set = frozenset(braille_motors)
    
    centroid_motor_history.append((current_time, centroid_set))
    braille_motor_history.append((current_time, braille_set))
    
    if len(braille_motor_history) < 3:
        return False, 0.0
    
    oldest_time = braille_motor_history[0][0]
    stable_duration = current_time - oldest_time
    
    if stable_duration < STABILITY_DURATION:
        return False, stable_duration
    
    recent_braille = [s for t, s in braille_motor_history 
                      if current_time - t <= STABILITY_DURATION]
    
    if len(recent_braille) < 3:
        return False, stable_duration
    
    current_braille = braille_set
    similarities = []
    
    for past_braille in recent_braille:
        if len(current_braille) == 0 or len(past_braille) == 0:
            similarities.append(0.0)
            continue
        
        intersection = len(current_braille & past_braille)
        union = len(current_braille | past_braille)
        
        if union == 0:
            similarity = 0.0
        else:
            similarity = intersection / union
        
        similarities.append(similarity)
    
    avg_similarity = np.mean(similarities) if similarities else 0.0
    is_stable = avg_similarity >= BRAILLE_SIMILARITY_THRESHOLD
    
    return is_stable, stable_duration


def display_stable_sequence(ax, frame, blurred, contour_cm, centroid_motors, braille_motors, 
                            braille_map, motor_coords, heel, toe):
    """Display the stable foot sequence with motor output."""
    global system_state, frozen_braille_motors, frozen_centroid_motors
    global frozen_braille_map, frozen_contour, frozen_heel, frozen_toe
    
    # OUTPUT: Complete motor configuration
    print("\n" + "="*60)
    print("STABLE CONFIGURATION DETECTED")
    print("="*60)
    
    print("\n=== CENTROID MOTORS (12 motors) ===")
    print_motor_output(centroid_motors, "Centroid Motor Indices")
    
    print("\n=== BRAILLE MOTORS (6 motors) ===")
    print_motor_output(braille_motors, "Braille Motor Indices")
    
    if braille_map:
        print("\nBraille Dot to Motor Mapping:")
        print("  Layout: 1 4")
        print("          2 5")
        print("          3 6")
        for dot in sorted(braille_map.keys()):
            print(f"  Dot {dot} → Motor {braille_map[dot]}")
    
    word = "DANGER"
    print("\n=== DANGER SEQUENCE ===")
    print(f"Word: {word}")
    print("Letters and their braille patterns:")
    for letter in word:
        dots = BRAILLE_PATTERNS.get(letter, [])
        motors = [braille_map[dot] for dot in dots if dot in braille_map] if braille_map else []
        print(f"  {letter}: dots {dots} → motors {motors}")
    
    print("\n" + "="*60 + "\n")
    
    # === STAGE 1: Show both centroid and braille motors for 1 second ===
    print("Displaying motors for 1 second...")
    
    num_frames = int(1.0 / REFRESH_DELAY)
    
    for _ in range(num_frames):
        ax.clear()
        ax.set_facecolor('black')
        
        ax.imshow(blurred, cmap='inferno', vmin=0, vmax=255, origin='upper',
                 extent=[0, GRID_SIZE_CM, GRID_SIZE_CM, 0], interpolation='bilinear')
        
        motor_x, motor_y = zip(*MOTOR_GRID)
        ax.scatter(motor_x, motor_y, s=25, color='gray', alpha=0.5, zorder=2)
        
        ax.plot(contour_cm[:, 0], contour_cm[:, 1], color='cyan', lw=2.5, zorder=3)
        
        braille_motor_coords = motor_coords[braille_motors]
        ax.scatter(braille_motor_coords[:, 0], braille_motor_coords[:, 1],
                  s=180, color='lime', marker='o',
                  edgecolors='white', lw=2.5, zorder=6, label='Braille Motors')
        
        centroid_motor_coords = motor_coords[centroid_motors]
        ax.scatter(centroid_motor_coords[:, 0], centroid_motor_coords[:, 1],
                  s=80, color='red', marker='o',
                  edgecolors='white', lw=1.5, zorder=7, label='Centroid Motors')
        
        ax.scatter(*heel, color='blue', s=70, edgecolors='white', lw=1.5, zorder=5)
        ax.scatter(*toe, color='red', s=70, edgecolors='white', lw=1.5, zorder=5)
        
        ax.set_xlim(0, GRID_SIZE_CM)
        ax.set_ylim(GRID_SIZE_CM, 0)
        ax.set_title("STABLE FOOT - Showing Motors", color='white', fontsize=13, weight='bold')
        ax.set_xlabel("Width (cm)", color='white')
        ax.set_ylabel("Height (cm)", color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
        
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        plt.pause(REFRESH_DELAY)
    
    # === STAGE 2: Display "DANGER" word letter by letter ===
    print("\nDisplaying DANGER in braille...\n")
    
    for letter_idx, letter in enumerate(word):
        active_motors = get_braille_motors_for_letter(letter, braille_map)
        
        if not active_motors:
            print(f"Warning: Could not map letter '{letter}' to motors")
            continue
        
        # OUTPUT: Letter and its motors
        print(f"=== DISPLAYING LETTER: {letter} ({letter_idx + 1}/{len(word)}) ===")
        print_motor_output(active_motors, f"Active Motors for '{letter}'")
        braille_dots = BRAILLE_PATTERNS.get(letter, [])
        print(f"Braille Dots: {braille_dots}")
        print(f"Motor Mapping: {dict((dot, braille_map[dot]) for dot in braille_dots if dot in braille_map)}")
        print()
        
        interpolation_steps = int(LETTER_DISPLAY_DURATION / REFRESH_DELAY)
        
        for step in range(interpolation_steps):
            t = step / (interpolation_steps - 1) if interpolation_steps > 1 else 0
            
            r = 0.75 + t * 0.25
            g = 1.0
            b = 0.0 + t * 1.0
            active_color = (r, g, b)
            
            ax.clear()
            ax.set_facecolor('black')
            
            ax.imshow(blurred, cmap='inferno', vmin=0, vmax=255, origin='upper',
                     extent=[0, GRID_SIZE_CM, GRID_SIZE_CM, 0], interpolation='bilinear')
            
            motor_x, motor_y = zip(*MOTOR_GRID)
            ax.scatter(motor_x, motor_y, s=25, color='gray', alpha=0.5, zorder=2)
            
            ax.plot(contour_cm[:, 0], contour_cm[:, 1], color='cyan', lw=2.5, zorder=3)
            
            inactive_motors = [m for m in braille_motors if m not in active_motors]
            if inactive_motors:
                inactive_motor_coords = motor_coords[inactive_motors]
                ax.scatter(inactive_motor_coords[:, 0], inactive_motor_coords[:, 1],
                          s=180, color='lime', marker='o',
                          edgecolors='white', lw=2.5, zorder=6, alpha=0.6, label='Inactive Motors')
            
            if active_motors:
                active_motor_coords = motor_coords[active_motors]
                ax.scatter(active_motor_coords[:, 0], active_motor_coords[:, 1],
                          s=180, color=active_color, marker='o',
                          edgecolors='white', lw=2.5, zorder=7, label=f'Active: {letter}')
            
            ax.scatter(*heel, color='blue', s=70, edgecolors='white', lw=1.5, zorder=5)
            ax.scatter(*toe, color='red', s=70, edgecolors='white', lw=1.5, zorder=5)
            
            ax.set_xlim(0, GRID_SIZE_CM)
            ax.set_ylim(GRID_SIZE_CM, 0)
            ax.set_title(f"DANGER - Letter: {letter} ({letter_idx + 1}/{len(word)})", 
                        color='white', fontsize=13, weight='bold')
            ax.set_xlabel("Width (cm)", color='white')
            ax.set_ylabel("Height (cm)", color='white')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
            
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
            plt.pause(REFRESH_DELAY)
        
        if letter_idx < len(word) - 1:
            time.sleep(INTER_LETTER_WAIT)
    
    print("\nDANGER sequence complete - Entering FROZEN state (waiting for foot lift)...\n")
    
    # === STAGE 3: Enter FROZEN state ===
    system_state = "FROZEN"
    frozen_braille_motors = braille_motors
    frozen_centroid_motors = centroid_motors
    frozen_braille_map = braille_map
    frozen_contour = contour_cm
    frozen_heel = heel
    frozen_toe = toe
    
    centroid_motor_history.clear()
    braille_motor_history.clear()


def compute_medial_axis_bisector(contour, pressure_map):
    """Compute bisector using medial axis"""
    xy_px = contour[:, ::-1]
    h, w = pressure_map.shape
    mask = np.zeros((h, w), dtype=bool)
    contour_int = contour.astype(int)
    contour_int[:, 0] = np.clip(contour_int[:, 0], 0, h - 1)
    contour_int[:, 1] = np.clip(contour_int[:, 1], 0, w - 1)
    
    for i in range(len(contour_int)):
        mask[contour_int[i, 0], contour_int[i, 1]] = True
    mask = binary_fill_holes(mask)
    
    skeleton = morphology.skeletonize(mask)
    skeleton_points = np.argwhere(skeleton)
    
    if len(skeleton_points) < 5:
        xy_cm = xy_px * CM_PER_PIXEL
        mean = np.mean(xy_cm, axis=0)
        cov = np.cov((xy_cm - mean).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, np.argmax(eigvals)]
        axis /= np.linalg.norm(axis)
        proj = np.dot(xy_cm - mean, axis)
        heel_point = xy_cm[np.argmin(proj)]
        toe_point = xy_cm[np.argmax(proj)]
        return heel_point, toe_point, axis, mean
    
    skeleton_xy = skeleton_points[:, ::-1]
    skeleton_mean = np.mean(skeleton_xy, axis=0)
    skeleton_cov = np.cov((skeleton_xy - skeleton_mean).T)
    skel_eigvals, skel_eigvecs = np.linalg.eigh(skeleton_cov)
    skeleton_axis = skel_eigvecs[:, np.argmax(skel_eigvals)]
    skeleton_axis /= np.linalg.norm(skeleton_axis)
    
    skeleton_mean_cm = skeleton_mean * CM_PER_PIXEL
    xy_cm = xy_px * CM_PER_PIXEL
    proj = np.dot(xy_cm - skeleton_mean_cm, skeleton_axis)
    heel_point = xy_cm[np.argmin(proj)]
    toe_point = xy_cm[np.argmax(proj)]
    
    return heel_point, toe_point, skeleton_axis, skeleton_mean_cm


def compute_equal_area_spline(contour_cm, bisector_axis, center_cm):
    """Compute SMOOTH geometric spline that splits foot into equal areas."""
    proj_along = np.dot(contour_cm - center_cm, bisector_axis)
    min_proj = np.min(proj_along)
    max_proj = np.max(proj_along)
    
    num_samples = 15
    t_values = np.linspace(min_proj, max_proj, num_samples)
    
    perp = np.array([-bisector_axis[1], bisector_axis[0]])
    spline_points = []
    path = Path(contour_cm)
    
    for t in t_values:
        base_point = center_cm + bisector_axis * t
        
        left_edge = None
        right_edge = None
        
        for dist in np.linspace(0, 15, 60):
            test_left = base_point - perp * dist
            test_right = base_point + perp * dist
            
            if left_edge is None and not path.contains_point(test_left) and dist > 0:
                left_edge = base_point - perp * (dist - 15/60)
            if right_edge is None and not path.contains_point(test_right) and dist > 0:
                right_edge = base_point + perp * (dist - 15/60)
            
            if left_edge is not None and right_edge is not None:
                break
        
        if left_edge is not None and right_edge is not None:
            midpoint = (left_edge + right_edge) / 2
            spline_points.append(midpoint)
        else:
            spline_points.append(base_point)
    
    if len(spline_points) < 4:
        return {
            'spline_x': lambda t: center_cm[0] + bisector_axis[0] * t,
            'spline_y': lambda t: center_cm[1] + bisector_axis[1] * t,
            't_min': min_proj,
            't_max': max_proj,
            'points': np.array([center_cm + bisector_axis * t for t in t_values]),
            'is_fallback': True
        }
    
    spline_points = np.array(spline_points)
    t_param = np.dot(spline_points - spline_points[0], bisector_axis)
    
    try:
        spline_x = UnivariateSpline(t_param, spline_points[:, 0], s=2.0, k=2)
        spline_y = UnivariateSpline(t_param, spline_points[:, 1], s=2.0, k=2)
        
        return {
            'spline_x': spline_x,
            'spline_y': spline_y,
            't_min': np.min(t_param),
            't_max': np.max(t_param),
            'points': spline_points,
            'is_fallback': False
        }
    except:
        return {
            'spline_x': lambda t: center_cm[0] + bisector_axis[0] * t,
            'spline_y': lambda t: center_cm[1] + bisector_axis[1] * t,
            't_min': min_proj,
            't_max': max_proj,
            'points': np.array([center_cm + bisector_axis * t for t in t_values]),
            'is_fallback': True
        }


def find_motor_pairs_along_spline(motor_coords, spline_data, contour_cm, bisector_axis):
    """Find valid motor pairs - STRICT requirements for dice pattern."""
    if spline_data is None:
        return []
    
    contour_center = np.mean(contour_cm, axis=0)
    expanded_contour = contour_center + (contour_cm - contour_center) * 1.15
    expanded_path = Path(expanded_contour)
    
    valid_motors = []
    for i, motor in enumerate(motor_coords):
        if expanded_path.contains_point(motor):
            valid_motors.append((i, motor))
        else:
            dist = np.min(np.linalg.norm(expanded_contour - motor, axis=1))
            if dist < 2.5:
                valid_motors.append((i, motor))
    
    if len(valid_motors) < 2:
        return []
    
    spline_x = spline_data['spline_x']
    spline_y = spline_data['spline_y']
    t_min = spline_data['t_min']
    t_max = spline_data['t_max']
    is_fallback = spline_data.get('is_fallback', False)
    
    t_samples = np.linspace(t_min, t_max, 100)
    spline_samples = np.array([spline_x(t_samples), spline_y(t_samples)]).T
    
    motor_sides = []
    for idx, motor in valid_motors:
        dists = np.linalg.norm(spline_samples - motor, axis=1)
        closest_idx = np.argmin(dists)
        closest_t = t_samples[closest_idx]
        
        if is_fallback:
            tangent = bisector_axis
        else:
            try:
                dx = spline_x.derivative()(closest_t)
                dy = spline_y.derivative()(closest_t)
                tangent = np.array([dx, dy])
                if np.linalg.norm(tangent) > 0:
                    tangent /= np.linalg.norm(tangent)
                else:
                    tangent = bisector_axis
            except:
                tangent = bisector_axis
        
        normal = np.array([-tangent[1], tangent[0]])
        spline_point = np.array([spline_x(closest_t), spline_y(closest_t)])
        to_motor = motor - spline_point
        side = np.dot(to_motor, normal)
        
        motor_sides.append({
            'idx': idx,
            'motor': motor,
            'spline_t': closest_t,
            'spline_point': spline_point,
            'normal': normal,
            'tangent': tangent,
            'side': side,
            'dist_to_spline': np.linalg.norm(to_motor)
        })
    
    pairs = []
    left_motors = [m for m in motor_sides if m['side'] < -0.3]
    right_motors = [m for m in motor_sides if m['side'] > 0.3]
    
    t_range = t_max - t_min
    
    for left in left_motors:
        for right in right_motors:
            t_diff = abs(left['spline_t'] - right['spline_t'])
            
            if t_diff < t_range * 0.08:
                avg_tangent = (left['tangent'] + right['tangent']) / 2
                if np.linalg.norm(avg_tangent) > 0:
                    avg_tangent /= np.linalg.norm(avg_tangent)
                else:
                    avg_tangent = bisector_axis
                
                pair_vec = right['motor'] - left['motor']
                pair_vec_norm = pair_vec / np.linalg.norm(pair_vec)
                
                perp_dot = abs(np.dot(pair_vec_norm, avg_tangent))
                perpendicularity = 1.0 - perp_dot
                
                if perpendicularity > 0.85:
                    avg_t = (left['spline_t'] + right['spline_t']) / 2
                    
                    left_dist = left['dist_to_spline']
                    right_dist = right['dist_to_spline']
                    symmetry = 1.0 - abs(left_dist - right_dist) / max(left_dist, right_dist, 1.0)
                    
                    pairs.append({
                        'left': left,
                        'right': right,
                        'avg_t': avg_t,
                        'perpendicularity': perpendicularity,
                        't_diff': t_diff,
                        'symmetry': symmetry
                    })
    
    return pairs


def evaluate_dice_pattern_fit(three_pairs, rectangle_angle, bisector_axis, contour_cm, prev_motor_indices=None):
    """Score how well 3 pairs match the ideal dice-6 pattern."""
    if len(three_pairs) != 3:
        return float('inf')
    
    rect_axis = np.array([np.cos(rectangle_angle), np.sin(rectangle_angle)])
    
    positions = []
    pair_centers = []
    all_motors = []
    for pair in three_pairs:
        left_motor = pair['left']['motor']
        right_motor = pair['right']['motor']
        avg_pos = (left_motor + right_motor) / 2
        positions.append(avg_pos)
        pair_centers.append(avg_pos)
        all_motors.extend([left_motor, right_motor])
    
    projs = [np.dot(pos, rect_axis) for pos in positions]
    sorted_indices = np.argsort(projs)
    sorted_pairs = [three_pairs[i] for i in sorted_indices]
    
    path = Path(contour_cm)
    motors_inside_count = 0
    motors_distance_penalty = 0
    
    for motor in all_motors:
        if path.contains_point(motor):
            motors_inside_count += 1
        else:
            distances = np.linalg.norm(contour_cm - motor, axis=1)
            min_dist = np.min(distances)
            motors_distance_penalty += min_dist ** 2
    
    inside_score = (6 - motors_inside_count) * 100.0 + motors_distance_penalty * 20.0
    
    stability_bonus = 0
    if prev_motor_indices is not None:
        current_indices = set()
        for pair in three_pairs:
            current_indices.add(pair['left']['idx'])
            current_indices.add(pair['right']['idx'])
        
        matches = len(current_indices.intersection(prev_motor_indices))
        if matches >= 4:
            stability_bonus = -20.0 * (matches / 6.0)
    
    min_proj = min(projs)
    max_proj = max(projs)
    span = max_proj - min_proj
    
    if span < 3.0:
        max_separation_score = 200.0
    else:
        separations = []
        for i in range(len(pair_centers)):
            for j in range(i + 1, len(pair_centers)):
                dist = np.linalg.norm(pair_centers[i] - pair_centers[j])
                separations.append(dist)
        
        min_separation = min(separations)
        avg_separation = np.mean(separations)
        
        if avg_separation > 7.0:
            max_separation_score = 0
        else:
            max_separation_score = (7.0 - avg_separation) ** 2 * 3.0
        
        if min_separation < 4.0:
            max_separation_score += (4.0 - min_separation) ** 2 * 2.0
    
    actual_positions = [(projs[i] - min_proj) / span for i in sorted_indices]
    
    ideal_pattern_1 = [0.15, 0.50, 0.85]
    ideal_pattern_2 = [0.20, 0.50, 0.80]
    
    position_score_1 = sum((ideal - actual) ** 2 for ideal, actual in zip(ideal_pattern_1, actual_positions))
    position_score_2 = sum((ideal - actual) ** 2 for ideal, actual in zip(ideal_pattern_2, actual_positions))
    
    position_score = min(position_score_1, position_score_2) * 10.0
    
    perp_score = 0
    for pair in sorted_pairs:
        perp_deviation = 1.0 - pair['perpendicularity']
        perp_score += perp_deviation ** 2
    
    pair_widths = []
    for pair in sorted_pairs:
        left_motor = pair['left']['motor']
        right_motor = pair['right']['motor']
        width = np.linalg.norm(right_motor - left_motor)
        pair_widths.append(width)
    
    avg_width = np.mean(pair_widths)
    width_variance = np.std(pair_widths)
    width_consistency_score = width_variance ** 2
    
    symmetry_score = 0
    for pair in sorted_pairs:
        symmetry_deviation = 1.0 - pair.get('symmetry', 0.5)
        symmetry_score += symmetry_deviation ** 2
    
    levelness_score = 0
    for pair in sorted_pairs:
        t_diff_penalty = pair['t_diff']
        levelness_score += t_diff_penalty ** 2
    
    if span < 8.0:
        coverage_penalty = (8.0 - span) * 1.5
    elif span > 18.0:
        coverage_penalty = (span - 18.0) * 0.5
    else:
        coverage_penalty = 0
    
    total_score = (
        inside_score * 100.0 +
        max_separation_score * 60.0 +
        position_score * 50.0 +
        perp_score * 40.0 +
        symmetry_score * 25.0 +
        levelness_score * 20.0 +
        width_consistency_score * 15.0 +
        coverage_penalty * 10.0 +
        stability_bonus
    )
    
    return total_score


def select_best_motor_configuration(motor_pairs, bisector_axis, center_cm, contour_cm, prev_motor_indices=None, prev_score=None):
    """Test multiple rectangle orientations and find best 3-pair configuration."""
    if len(motor_pairs) < 3:
        return None
    
    bisector_angle = np.arctan2(bisector_axis[1], bisector_axis[0])
    
    best_config = None
    best_score = float('inf')
    
    for angle_offset in range(-60, 65, 5):
        angle_deg = angle_offset
        angle_rad = bisector_angle + np.radians(angle_deg)
        
        rect_axis = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        
        pair_projections = []
        for pair in motor_pairs:
            left_motor = pair['left']['motor']
            right_motor = pair['right']['motor']
            avg_pos = (left_motor + right_motor) / 2
            proj = np.dot(avg_pos - center_cm, rect_axis)
            pair_projections.append((proj, pair))
        
        pair_projections.sort(key=lambda x: x[0])
        
        from itertools import combinations
        for combo in combinations(range(len(pair_projections)), 3):
            three_pairs = [pair_projections[i][1] for i in combo]
            
            motor_indices = set()
            motors_are_unique = True
            
            for pair in three_pairs:
                left_idx = pair['left']['idx']
                right_idx = pair['right']['idx']
                
                if left_idx in motor_indices or right_idx in motor_indices:
                    motors_are_unique = False
                    break
                
                if left_idx == right_idx:
                    motors_are_unique = False
                    break
                
                motor_indices.add(left_idx)
                motor_indices.add(right_idx)
            
            if not motors_are_unique or len(motor_indices) != 6:
                continue
            
            pair_centers = []
            for pair in three_pairs:
                left_motor = pair['left']['motor']
                right_motor = pair['right']['motor']
                center = (left_motor + right_motor) / 2
                pair_centers.append(center)
            
            min_pair_distance = float('inf')
            for i in range(len(pair_centers)):
                for j in range(i + 1, len(pair_centers)):
                    dist = np.linalg.norm(pair_centers[i] - pair_centers[j])
                    min_pair_distance = min(min_pair_distance, dist)
            
            if min_pair_distance < 3.5:
                continue
            
            score = evaluate_dice_pattern_fit(three_pairs, angle_rad, bisector_axis, contour_cm, prev_motor_indices)
            
            if score < best_score:
                best_score = score
                best_config = {
                    'pairs': three_pairs,
                    'angle': angle_rad,
                    'score': score,
                    'motor_indices': motor_indices,
                    'min_separation': min_pair_distance
                }
    
    if best_config is not None and prev_score is not None and prev_motor_indices is not None:
        improvement_ratio = (prev_score - best_score) / prev_score if prev_score > 0 else 1.0
        
        if improvement_ratio < 0.25:
            prev_still_valid = prev_motor_indices.issubset(
                set(m['idx'] for pair in motor_pairs for m in [pair['left'], pair['right']])
            )
            
            if prev_still_valid:
                for angle_offset in range(-60, 65, 5):
                    angle_rad = bisector_angle + np.radians(angle_offset)
                    rect_axis = np.array([np.cos(angle_rad), np.sin(angle_rad)])
                    
                    pair_projections = []
                    for pair in motor_pairs:
                        left_motor = pair['left']['motor']
                        right_motor = pair['right']['motor']
                        avg_pos = (left_motor + right_motor) / 2
                        proj = np.dot(avg_pos - center_cm, rect_axis)
                        pair_projections.append((proj, pair))
                    
                    from itertools import combinations
                    for combo in combinations(range(len(pair_projections)), 3):
                        three_pairs = [pair_projections[i][1] for i in combo]
                        
                        current_indices = set()
                        for pair in three_pairs:
                            current_indices.add(pair['left']['idx'])
                            current_indices.add(pair['right']['idx'])
                        
                        if current_indices == prev_motor_indices:
                            return {
                                'pairs': three_pairs,
                                'angle': angle_rad,
                                'score': prev_score,
                                'motor_indices': prev_motor_indices,
                                'min_separation': min([np.linalg.norm(
                                    (three_pairs[i]['left']['motor'] + three_pairs[i]['right']['motor'])/2 -
                                    (three_pairs[j]['left']['motor'] + three_pairs[j]['right']['motor'])/2
                                ) for i in range(3) for j in range(i+1, 3)])
                            }
    
    return best_config


def smooth_temporal(current_axis, current_heel, current_toe):
    """Temporal smoothing"""
    global prev_axis, prev_heel, prev_toe
    
    if prev_axis is None:
        prev_axis = current_axis
        prev_heel = current_heel
        prev_toe = current_toe
        return current_axis, current_heel, current_toe
    
    if np.dot(current_axis, prev_axis) < 0:
        current_axis = -current_axis
        current_heel, current_toe = current_toe, current_heel
    
    smoothed_axis = SMOOTHING_ALPHA * prev_axis + (1 - SMOOTHING_ALPHA) * current_axis
    smoothed_axis /= np.linalg.norm(smoothed_axis)
    smoothed_heel = SMOOTHING_ALPHA * prev_heel + (1 - SMOOTHING_ALPHA) * current_heel
    smoothed_toe = SMOOTHING_ALPHA * prev_toe + (1 - SMOOTHING_ALPHA) * current_toe
    
    prev_axis = smoothed_axis
    prev_heel = smoothed_heel
    prev_toe = smoothed_toe
    
    return smoothed_axis, smoothed_heel, smoothed_toe


# ============================================================
# --- MAIN LOOP ---
# ============================================================

try:
    while True:
        frame = read_frame()
        if frame is None:
            continue

        frame = frame.T
        frame = np.hstack((frame[:, 2:], frame[:, :2]))
        frame = np.clip(frame - baseline_mean, 0, 255)

        blurred = gaussian_filter(frame, sigma=SIGMA)
        
        if system_state == "FROZEN":
            foot_lifted = detect_foot_lift(blurred, frozen_contour)
            
            if foot_lifted:
                print("\nFOOT LIFTED - Resuming normal detection\n")
                system_state = "NORMAL"
                frozen_braille_motors = None
                frozen_centroid_motors = None
                frozen_braille_map = None
                frozen_contour = None
                frozen_heel = None
                frozen_toe = None
                centroid_motor_history.clear()
                braille_motor_history.clear()
                continue
            else:
                ax.clear()
                ax.set_facecolor('black')
                
                ax.imshow(blurred, cmap='inferno', vmin=0, vmax=255, origin='upper',
                         extent=[0, GRID_SIZE_CM, GRID_SIZE_CM, 0], interpolation='bilinear')
                
                motor_x, motor_y = zip(*MOTOR_GRID)
                ax.scatter(motor_x, motor_y, s=25, color='gray', alpha=0.5, zorder=2)
                
                if frozen_contour is not None:
                    ax.plot(frozen_contour[:, 0], frozen_contour[:, 1], color='cyan', lw=2.5, zorder=3)
                
                if frozen_braille_motors is not None:
                    braille_motor_coords = MOTOR_COORDS[frozen_braille_motors]
                    ax.scatter(braille_motor_coords[:, 0], braille_motor_coords[:, 1],
                              s=180, color='lime', marker='o',
                              edgecolors='white', lw=2.5, zorder=6, label='Braille Motors (Frozen)')
                
                if frozen_centroid_motors is not None:
                    centroid_motor_coords = MOTOR_COORDS[frozen_centroid_motors]
                    ax.scatter(centroid_motor_coords[:, 0], centroid_motor_coords[:, 1],
                              s=80, color='red', marker='o',
                              edgecolors='white', lw=1.5, zorder=7, alpha=0.7, label='Centroid Motors (Frozen)')
                
                if frozen_heel is not None and frozen_toe is not None:
                    ax.scatter(*frozen_heel, color='blue', s=70, edgecolors='white', lw=1.5, zorder=5)
                    ax.scatter(*frozen_toe, color='red', s=70, edgecolors='white', lw=1.5, zorder=5)
                
                ax.set_xlim(0, GRID_SIZE_CM)
                ax.set_ylim(GRID_SIZE_CM, 0)
                ax.set_title("FROZEN STATE - Lift Foot to Resume", color='yellow', fontsize=13, weight='bold')
                ax.set_xlabel("Width (cm)", color='white')
                ax.set_ylabel("Height (cm)", color='white')
                ax.tick_params(colors='white')
                for spine in ax.spines.values():
                    spine.set_color('white')
                
                ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
                plt.pause(REFRESH_DELAY)
                continue
        
        thresh = np.median(blurred) + 0.55 * (np.max(blurred) - np.median(blurred))
        mask = blurred > thresh

        ax.clear()
        ax.set_facecolor('black')
        im = ax.imshow(blurred, cmap='inferno', vmin=0, vmax=255, origin='upper',
                      extent=[0, GRID_SIZE_CM, GRID_SIZE_CM, 0], interpolation='bilinear')
        
        motor_x, motor_y = zip(*MOTOR_GRID)
        ax.scatter(motor_x, motor_y, s=25, color='gray', alpha=0.5, zorder=2)
        ax.set_xlim(0, GRID_SIZE_CM)
        ax.set_ylim(GRID_SIZE_CM, 0)
        ax.set_title("Geometric Spline Dice Pattern Selector", color='white', fontsize=13, weight='bold')
        ax.set_xlabel("Width (cm)", color='white')
        ax.set_ylabel("Height (cm)", color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')

        contours = measure.find_contours(mask, 0.5)
        if not contours:
            plt.pause(REFRESH_DELAY)
            continue

        contour = max(contours, key=lambda c: len(c))
        contour_cm = contour[:, ::-1] * CM_PER_PIXEL
        
        heel, toe, axis, center = compute_medial_axis_bisector(contour, blurred)
        axis, heel, toe = smooth_temporal(axis, heel, toe)
        
        heel, toe, axis = verify_and_correct_orientation(heel, toe, axis, contour_cm)
        
        ax.plot(contour_cm[:, 0], contour_cm[:, 1], color='cyan', lw=2.5, zorder=3)
        
        spline_data = compute_equal_area_spline(contour_cm, axis, center)
        
        foot_centroid = compute_foot_centroid(contour_cm)
        centroid_motors = find_nearest_motors(foot_centroid, MOTOR_COORDS, n=12)
        
        braille_motors = []
        braille_map = None
        
        if spline_data is not None:
            t_range = np.linspace(spline_data['t_min'], spline_data['t_max'], 100)
            spline_curve = np.array([spline_data['spline_x'](t_range), 
                                    spline_data['spline_y'](t_range)]).T
            ax.plot(spline_curve[:, 0], spline_curve[:, 1], 
                   'g-', lw=2.5, alpha=0.8, zorder=4, label='Equal-Area Spline')
            
            motor_pairs = find_motor_pairs_along_spline(MOTOR_COORDS, spline_data, contour_cm, axis)
            
            if len(motor_pairs) >= 3:
                best_config = select_best_motor_configuration(
                    motor_pairs, axis, center, contour_cm, prev_motor_indices, prev_config_score)
                
                if best_config is not None:
                    prev_motor_indices = best_config['motor_indices']
                    prev_config_score = best_config['score']
                    
                    braille_motors = list(best_config['motor_indices'])
                    braille_map = map_motors_to_braille_dots(braille_motors, MOTOR_COORDS, axis)
                    
                    is_stable, stable_duration = check_stability(centroid_motors, braille_motors)
                    
                    if is_stable:
                        display_stable_sequence(ax, frame, blurred, contour_cm, 
                                              centroid_motors, braille_motors, 
                                              braille_map, MOTOR_COORDS, heel, toe)
                        continue
                    
                    # OUTPUT: Both centroid and braille motors selected (only when braille motors are detected)
                    print("\n" + "="*50)
                    print("MOTOR SELECTION UPDATE")
                    print("="*50)
                    
                    print("\n=== CENTROID MOTORS (12 motors) ===")
                    print_motor_output(centroid_motors, "Centroid Motor Indices")
                    
                    print("\n=== BRAILLE MOTORS (6 motors) ===")
                    print_motor_output(braille_motors, "Braille Motor Indices")
                    if braille_map:
                        print("\nBraille Dot Mapping:")
                        for dot in sorted(braille_map.keys()):
                            print(f"  Dot {dot} → Motor {braille_map[dot]}")
                    print("\n" + "="*50 + "\n")
                    
                    selected_motors = []
                    for pair in best_config['pairs']:
                        left_motor = pair['left']['motor']
                        right_motor = pair['right']['motor']
                        
                        ax.plot([left_motor[0], right_motor[0]],
                               [left_motor[1], right_motor[1]],
                               color='yellow', ls='-', lw=2, alpha=0.8, zorder=5)
                        
                        selected_motors.extend([left_motor, right_motor])
                    
                    selected_motors = np.array(selected_motors)
                    ax.scatter(selected_motors[:, 0], selected_motors[:, 1],
                              s=180, color='lime', marker='o',
                              edgecolors='white', lw=2.5, zorder=6)
        
        # Draw centroid motors (no output here to reduce spam)
        centroid_motor_coords = MOTOR_COORDS[centroid_motors]
        ax.scatter(centroid_motor_coords[:, 0], centroid_motor_coords[:, 1],
                  s=80, color='red', marker='o',
                  edgecolors='white', lw=1.5, zorder=7, alpha=0.7)
        
        ax.scatter(*foot_centroid, color='cyan', s=50, marker='x', 
                  edgecolors='white', lw=2, zorder=8)
        
        ax.scatter(*heel, color='blue', s=70, edgecolors='white', lw=1.5, zorder=6)
        ax.scatter(*toe, color='red', s=70, edgecolors='white', lw=1.5, zorder=6)
        
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        plt.pause(REFRESH_DELAY)

except KeyboardInterrupt:
    print("\nExiting viewer.")
    ser.close()