# ============================================================
# Geometric Spline-Based 6-Motor Dice Pattern Selector
# ------------------------------------------------------------
# Advanced multi-stage algorithm:
# 1. Medial axis for bisector direction
# 2. Equal-area geometric spline splitting foot in half
# 3. Find all valid motor pairs perpendicular to spline
# 4. Test multiple rectangular orientations (-45° to +45°)
# 5. Select 3 pairs that best match dice-6 pattern
# ============================================================

import serial, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_fill_holes
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.integrate import quad
from skimage import measure, morphology
from matplotlib.path import Path

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
prev_motor_indices = None  # Track previous motor selection
prev_config_score = None   # Track previous configuration quality
SMOOTHING_ALPHA = 0.6
MOTOR_SWITCH_THRESHOLD = 0.75  # Only switch motors if new config is 25% better

# ============================================================
# --- FUNCTIONS ---
# ============================================================

def read_frame():
    ser.write(b'A')
    data = ser.read(GRID_SIZE * GRID_SIZE)
    if len(data) != GRID_SIZE * GRID_SIZE:
        return None
    return np.frombuffer(data, dtype=np.uint8).reshape((GRID_SIZE, GRID_SIZE))


def compute_medial_axis_bisector(contour, pressure_map):
    """Compute bisector using medial axis - already implemented"""
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
    """
    Compute SMOOTH geometric spline that splits foot into equal areas.
    Uses heavy smoothing to avoid wiggly curves that break perpendicularity.
    """
    # Project contour onto bisector to parameterize
    proj_along = np.dot(contour_cm - center_cm, bisector_axis)
    min_proj = np.min(proj_along)
    max_proj = np.max(proj_along)
    
    # Sample fewer points for smoother curve
    num_samples = 15  # Reduced from 25
    t_values = np.linspace(min_proj, max_proj, num_samples)
    
    perp = np.array([-bisector_axis[1], bisector_axis[0]])
    spline_points = []
    path = Path(contour_cm)
    
    for t in t_values:
        base_point = center_cm + bisector_axis * t
        
        # Find left and right edges
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
        # FALLBACK: return simple bisector line
        return {
            'spline_x': lambda t: center_cm[0] + bisector_axis[0] * t,
            'spline_y': lambda t: center_cm[1] + bisector_axis[1] * t,
            't_min': min_proj,
            't_max': max_proj,
            'points': np.array([center_cm + bisector_axis * t for t in t_values]),
            'is_fallback': True
        }
    
    spline_points = np.array(spline_points)
    
    # Project onto bisector for parameterization
    t_param = np.dot(spline_points - spline_points[0], bisector_axis)
    
    try:
        # HEAVY smoothing to avoid wiggles (s=2.0 instead of 0.5)
        # Lower degree for smoother curve (k=2 instead of 3)
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
        # FALLBACK: return simple bisector line
        return {
            'spline_x': lambda t: center_cm[0] + bisector_axis[0] * t,
            'spline_y': lambda t: center_cm[1] + bisector_axis[1] * t,
            't_min': min_proj,
            't_max': max_proj,
            'points': np.array([center_cm + bisector_axis * t for t in t_values]),
            'is_fallback': True
        }


def find_motor_pairs_along_spline(motor_coords, spline_data, contour_cm, bisector_axis):
    """
    Find valid motor pairs - STRICT requirements for dice pattern.
    Pairs must be truly perpendicular and at same spline position.
    """
    if spline_data is None:
        return []
    
    # Expand contour by 15%
    contour_center = np.mean(contour_cm, axis=0)
    expanded_contour = contour_center + (contour_cm - contour_center) * 1.15
    expanded_path = Path(expanded_contour)
    
    # Filter motors
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
    
    # Sample spline
    t_samples = np.linspace(t_min, t_max, 100)
    spline_samples = np.array([spline_x(t_samples), spline_y(t_samples)]).T
    
    # Classify motors by side of spline
    motor_sides = []
    for idx, motor in valid_motors:
        dists = np.linalg.norm(spline_samples - motor, axis=1)
        closest_idx = np.argmin(dists)
        closest_t = t_samples[closest_idx]
        
        # Get tangent at this point
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
    
    # Find pairs - STRICTER matching
    pairs = []
    left_motors = [m for m in motor_sides if m['side'] < -0.3]  # Must be clearly left
    right_motors = [m for m in motor_sides if m['side'] > 0.3]  # Must be clearly right
    
    t_range = t_max - t_min
    
    for left in left_motors:
        for right in right_motors:
            # STRICT: must be at very similar position (5% instead of 15%)
            t_diff = abs(left['spline_t'] - right['spline_t'])
            
            if t_diff < t_range * 0.08:  # Tighter: 8% tolerance
                # Check perpendicularity to AVERAGE tangent
                avg_tangent = (left['tangent'] + right['tangent']) / 2
                if np.linalg.norm(avg_tangent) > 0:
                    avg_tangent /= np.linalg.norm(avg_tangent)
                else:
                    avg_tangent = bisector_axis
                
                pair_vec = right['motor'] - left['motor']
                pair_vec_norm = pair_vec / np.linalg.norm(pair_vec)
                
                # Perpendicularity: dot product should be near 0
                # We want pair_vec perpendicular to tangent
                perp_dot = abs(np.dot(pair_vec_norm, avg_tangent))
                perpendicularity = 1.0 - perp_dot  # Close to 1 = very perpendicular
                
                if perpendicularity > 0.85:  # STRICT: 85% perpendicular (< 30° off)
                    avg_t = (left['spline_t'] + right['spline_t']) / 2
                    
                    # Symmetry check
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
    """
    Score how well 3 pairs match the ideal dice-6 pattern.
    PRIORITIZE: motors inside foot, max separation (heel/mid/toe), perpendicularity, stability.
    Lower score is better.
    """
    if len(three_pairs) != 3:
        return float('inf')
    
    # Sort pairs by position along rectangle
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
    
    # 0. MOTORS INSIDE FOOT CONTOUR (CRITICAL - highest priority)
    path = Path(contour_cm)
    motors_inside_count = 0
    motors_distance_penalty = 0
    
    for motor in all_motors:
        if path.contains_point(motor):
            motors_inside_count += 1
        else:
            # Calculate distance to nearest contour point
            distances = np.linalg.norm(contour_cm - motor, axis=1)
            min_dist = np.min(distances)
            motors_distance_penalty += min_dist ** 2  # Quadratic penalty for distance
    
    # Score based on how many motors are inside
    # Perfect: all 6 inside, Bad: motors outside
    inside_score = (6 - motors_inside_count) * 100.0 + motors_distance_penalty * 20.0
    
    # 1. STABILITY BONUS (prefer keeping same motors if they're still good)
    stability_bonus = 0
    if prev_motor_indices is not None:
        current_indices = set()
        for pair in three_pairs:
            current_indices.add(pair['left']['idx'])
            current_indices.add(pair['right']['idx'])
        
        # Count how many motors match previous selection
        matches = len(current_indices.intersection(prev_motor_indices))
        if matches >= 4:  # If at least 4 motors are same
            stability_bonus = -20.0 * (matches / 6.0)  # Bonus for stability
    
    # 2. MAXIMUM SEPARATION (prioritize heel/mid/toe distribution)
    min_proj = min(projs)
    max_proj = max(projs)
    span = max_proj - min_proj
    
    if span < 3.0:  # Too clustered
        max_separation_score = 200.0  # Heavy penalty
    else:
        # Calculate actual pairwise separations
        separations = []
        for i in range(len(pair_centers)):
            for j in range(i + 1, len(pair_centers)):
                dist = np.linalg.norm(pair_centers[i] - pair_centers[j])
                separations.append(dist)
        
        min_separation = min(separations)
        avg_separation = np.mean(separations)
        
        # Reward larger separations (want pairs spread out)
        # Ideal is ~8-10cm between pairs
        if avg_separation > 7.0:
            max_separation_score = 0  # Perfect
        else:
            max_separation_score = (7.0 - avg_separation) ** 2 * 3.0
        
        # Extra penalty if minimum separation is too small
        if min_separation < 4.0:
            max_separation_score += (4.0 - min_separation) ** 2 * 2.0
    
    # 3. HEEL/MID/TOE POSITIONING (ideal 25%, 50%, 75% or close to 0%, 50%, 100%)
    actual_positions = [(projs[i] - min_proj) / span for i in sorted_indices]
    
    # Try both ideal patterns and pick best
    ideal_pattern_1 = [0.15, 0.50, 0.85]  # 15%, 50%, 85% (heel/mid/toe with margin)
    ideal_pattern_2 = [0.20, 0.50, 0.80]  # 20%, 50%, 80% (alternative)
    
    position_score_1 = sum((ideal - actual) ** 2 for ideal, actual in zip(ideal_pattern_1, actual_positions))
    position_score_2 = sum((ideal - actual) ** 2 for ideal, actual in zip(ideal_pattern_2, actual_positions))
    
    position_score = min(position_score_1, position_score_2) * 10.0
    
    # 4. PERPENDICULARITY SCORE
    perp_score = 0
    for pair in sorted_pairs:
        perp_deviation = 1.0 - pair['perpendicularity']
        perp_score += perp_deviation ** 2
    
    # 5. PAIR WIDTH CONSISTENCY (all pairs should have similar width)
    pair_widths = []
    for pair in sorted_pairs:
        left_motor = pair['left']['motor']
        right_motor = pair['right']['motor']
        width = np.linalg.norm(right_motor - left_motor)
        pair_widths.append(width)
    
    avg_width = np.mean(pair_widths)
    width_variance = np.std(pair_widths)
    width_consistency_score = width_variance ** 2
    
    # 6. SYMMETRY SCORE (motors equidistant from spline)
    symmetry_score = 0
    for pair in sorted_pairs:
        symmetry_deviation = 1.0 - pair.get('symmetry', 0.5)
        symmetry_score += symmetry_deviation ** 2
    
    # 7. LEVELNESS SCORE (pairs at same height)
    levelness_score = 0
    for pair in sorted_pairs:
        t_diff_penalty = pair['t_diff']
        levelness_score += t_diff_penalty ** 2
    
    # 8. COVERAGE (span should cover good portion of foot)
    if span < 8.0:
        coverage_penalty = (8.0 - span) * 1.5
    elif span > 18.0:
        coverage_penalty = (span - 18.0) * 0.5  # Slight penalty for too spread
    else:
        coverage_penalty = 0
    
    # WEIGHTED COMBINATION - Inside foot is TOP priority
    total_score = (
        inside_score * 100.0 +           # HIGHEST: motors must be inside foot!
        max_separation_score * 60.0 +    # CRITICAL: maximize pair separation
        position_score * 50.0 +          # CRITICAL: heel/mid/toe positioning
        perp_score * 40.0 +              # IMPORTANT: perpendicularity
        symmetry_score * 25.0 +          # IMPORTANT: symmetry
        levelness_score * 20.0 +         # MODERATE: level pairs
        width_consistency_score * 15.0 + # MODERATE: uniform widths
        coverage_penalty * 10.0 +        # LOW: coverage
        stability_bonus                   # BONUS: prefer stable selections
    )
    
    return total_score


def select_best_motor_configuration(motor_pairs, bisector_axis, center_cm, prev_motor_indices=None, prev_score=None):
    """
    Test multiple rectangle orientations and find best 3-pair configuration.
    EXPANDED: -60° to +60° in 5° increments to handle extreme tilts.
    ENFORCES: Minimum pair separation, all 6 motors unique.
    STABLE: Prefers keeping same motors unless significantly better option exists.
    """
    if len(motor_pairs) < 3:
        return None
    
    # Bisector angle
    bisector_angle = np.arctan2(bisector_axis[1], bisector_axis[0])
    
    # Test different rectangle orientations - EXPANDED RANGE
    best_config = None
    best_score = float('inf')
    
    # Test from -60° to +60° to handle extreme tilts
    for angle_offset in range(-60, 65, 5):
        angle_deg = angle_offset
        angle_rad = bisector_angle + np.radians(angle_deg)
        
        rect_axis = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        
        # Project all pairs onto this rectangle axis
        pair_projections = []
        for pair in motor_pairs:
            left_motor = pair['left']['motor']
            right_motor = pair['right']['motor']
            avg_pos = (left_motor + right_motor) / 2
            proj = np.dot(avg_pos - center_cm, rect_axis)
            pair_projections.append((proj, pair))
        
        # Sort by projection
        pair_projections.sort(key=lambda x: x[0])
        
        # Try all combinations of 3 pairs
        from itertools import combinations
        for combo in combinations(range(len(pair_projections)), 3):
            three_pairs = [pair_projections[i][1] for i in combo]
            
            # === ENFORCE UNIQUE MOTORS ===
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
            
            # === ENFORCE MINIMUM PAIR SEPARATION ===
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
            
            # Require at least 3.5cm separation between pair centers
            if min_pair_distance < 3.5:
                continue
            
            # Evaluate this configuration (pass contour for inside check)
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
    
    # === STABILITY CHECK: Only switch if new config is significantly better ===
    if best_config is not None and prev_score is not None and prev_motor_indices is not None:
        # Check if we should keep the previous configuration
        # Only switch if new score is at least 25% better
        improvement_ratio = (prev_score - best_score) / prev_score if prev_score > 0 else 1.0
        
        if improvement_ratio < 0.25:  # Less than 25% improvement
            # Check if previous motors are still valid in current frame
            prev_still_valid = prev_motor_indices.issubset(
                set(m['idx'] for pair in motor_pairs for m in [pair['left'], pair['right']])
            )
            
            if prev_still_valid:
                # Keep previous configuration (find it in current options)
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
                        
                        # If this matches previous selection, use it
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
        
        # Step 1: Compute bisector
        heel, toe, axis, center = compute_medial_axis_bisector(contour, blurred)
        axis, heel, toe = smooth_temporal(axis, heel, toe)
        
        # Draw contour
        ax.plot(contour_cm[:, 0], contour_cm[:, 1], color='cyan', lw=2.5, zorder=3)
        
        # Step 2: Compute equal-area spline
        spline_data = compute_equal_area_spline(contour_cm, axis, center)
        
        if spline_data is not None:
            # Draw spline
            t_range = np.linspace(spline_data['t_min'], spline_data['t_max'], 100)
            spline_curve = np.array([spline_data['spline_x'](t_range), 
                                    spline_data['spline_y'](t_range)]).T
            ax.plot(spline_curve[:, 0], spline_curve[:, 1], 
                   'g-', lw=2.5, alpha=0.8, zorder=4, label='Equal-Area Spline')
            
            # Step 3: Find all motor pairs
            motor_pairs = find_motor_pairs_along_spline(MOTOR_COORDS, spline_data, contour_cm, axis)
            
            # Step 4: Select best 3-pair configuration
            if len(motor_pairs) >= 3:
                best_config = select_best_motor_configuration(
                    motor_pairs, axis, center, prev_motor_indices, prev_config_score)
                
                if best_config is not None:
                    # Update previous state for stability
                    prev_motor_indices = best_config['motor_indices']
                    prev_config_score = best_config['score']
                    
                    # Draw selected pairs
                    selected_motors = []
                    for pair in best_config['pairs']:
                        left_motor = pair['left']['motor']
                        right_motor = pair['right']['motor']
                        
                        # Draw connection line
                        ax.plot([left_motor[0], right_motor[0]],
                               [left_motor[1], right_motor[1]],
                               color='yellow', ls='-', lw=2, alpha=0.8, zorder=5)
                        
                        selected_motors.extend([left_motor, right_motor])
                    
                    # Highlight motors
                    selected_motors = np.array(selected_motors)
                    ax.scatter(selected_motors[:, 0], selected_motors[:, 1],
                              s=180, color='lime', marker='o',
                              edgecolors='white', lw=2.5, zorder=7)
        
        # Mark heel and toe
        ax.scatter(*heel, color='blue', s=70, edgecolors='white', lw=1.5, zorder=6)
        ax.scatter(*toe, color='red', s=70, edgecolors='white', lw=1.5, zorder=6)
        
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        plt.pause(REFRESH_DELAY)

except KeyboardInterrupt:
    print("\nExiting viewer.")
    ser.close()