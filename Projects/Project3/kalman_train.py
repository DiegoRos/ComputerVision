import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from kalman_dataset import PitchDataset
from kalman_filter import PitchKalmanFilter
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# --- CONFIG ---
root_path = Path.cwd()
competition_path = root_path / "competition_folder/baseball-pitch-tracking/kaggle_dataset"
data_path = competition_path / "data"

train_video_dir = competition_path / "train_trimmed"
train_csv_path = data_path / "train_ground_truth.csv"

VIDEO_DIR = str(train_video_dir)
CSV_PATH = str(train_csv_path)
YOLO_PATH = "./competition_folder/model_output/ball_finetune/weights/best.pt"

COMPETITION_ROOT = root_path / "competition_folder/baseball-pitch-tracking/kaggle_dataset"

TEST_VIDEO_DIR = COMPETITION_ROOT / "test"
TEST_CSV_PATH = COMPETITION_ROOT / "data/test_features.csv"

# --- CONSTANTS FOR ZONE DEFINITION ---
# Plate is 17 inches wide.
# 17 inches / 12 inches/ft = 1.41666... ft
PLATE_WIDTH_FT = 17 / 12  
PLATE_HALF_WIDTH_FT = 17 / 24 # 0.70833... ft

# Ball radius approx 1.45 inches ~ 1.57 inches? 
# Standard baseball diameter is 2.86-2.94 inches. Radius ~ 1.45 inches.
BALL_RADIUS_FT = 1.45 / 12 

# For training the model we want to create a train/val/test split with a 80/20 spilt
def create_data_loaders(csv_path, video_dir, yolo_model_dir, shuffle=True):
    # Load csv
    df = pd.read_csv(str(csv_path))
    # Create first split (70/30) spliting into train and val/test
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42, # For reproducibility
        stratify=df['pitch_class'] # Ensure that we get a smiliar distribution of ball/strike in testing set
    )

    print(f"Full Dataset: {len(df)}")
    print(f"Train Set:    {len(train_df)} ({(len(train_df)/len(df))*100:.1f}%)")
    print(f"Val Set:      {len(val_df)} ({(len(val_df)/len(df))*100:.1f}%)")

    # Create datasets
    train_ds = PitchDataset(train_df, video_dir, yolo_model_dir)
    val_ds = PitchDataset(val_df, video_dir, yolo_model_dir) # We will still calculate loss of these so we send them in training mod

    return train_ds, val_ds
# --- FEATURE EXTRACTION & MODEL TRAINING ---
# 1. Feature Extraction Loop
def extract_features(dataset, desc="Extracting Features"):
    features = []
    targets_x = []
    targets_z = []
    meta_labels = []

    print(f"{desc}...")
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        
        traj = data['trajectory']
        plate_loc = data['plate_loc']
        px_per_ft = data['px_per_ft']
        phys = data['physics']
        
        # Skip empty videos
        if len(traj) == 0:
            continue
            
        # --- A. Kalman Filter Processing ---
        # We keep dt=1.0 (Time Step = 1 Frame) for numerical stability.
        kf = PitchKalmanFilter(dt=1.0)
        final_state = kf.process_sequence(traj)
        
        # State: [x, y, vx, vy, ax, ay]
        last_x, last_y = final_state[0], final_state[1]
        last_vx_frame, last_vy_frame = final_state[2], final_state[3]
        
        # --- B. Time of Flight Calculations ---
        release_ext = phys.get('release_extension', 6.0)
        release_speed = phys.get('release_speed', 90.0)
        
        tof_total = (60.5 - release_ext) / (release_speed * 1.4667)
        
        # Frames used
        n_frames = len(traj)
        
        # Estimated time per frame (dt_real) in seconds
        current_flight_time = tof_total * 0.8
        dt_real = current_flight_time / n_frames if n_frames > 0 else 0.016
        
        # Remaining time
        remaining_time = tof_total * 0.2
        remaining_frames = remaining_time / dt_real

        # --- C. Normalize to Plate Context & Convert Units ---
        # Position relative to plate center (in feet)
        rel_x_ft = (last_x - plate_loc[0]) / px_per_ft
        rel_y_ft = (last_y - plate_loc[1]) / px_per_ft 
        
        # Velocity Conversion to Feet/Second
        vel_x_fps = (last_vx_frame / dt_real) / px_per_ft
        vel_y_fps = (last_vy_frame / dt_real) / px_per_ft
        
        # --- D. Analytical Physics Prediction (The "Prior") ---
        pfx_x_ft = phys.get('pfx_x', 0) / 12.0
        pfx_z_ft = phys.get('pfx_z', 0) / 12.0
        
        phys_pred_x = phys.get('release_pos_x', 0) + (pfx_x_ft * (tof_total**2)) 
        phys_pred_z = phys.get('release_pos_z', 0) - (16.1 * (tof_total**2)) + (pfx_z_ft * (tof_total**2))

        # --- E. Compile Feature Vector ---
        row_feat = {
            # Visual State
            'vis_x_ft': rel_x_ft,
            'vis_z_ft': -rel_y_ft, 
            'vis_vx_fps': vel_x_fps,
            'vis_vz_fps': -vel_y_fps,
            
            # Physics Inputs
            'release_speed': release_speed,
            'release_spin': phys.get('release_spin_rate', 0),
            'pfx_x': phys.get('pfx_x', 0),
            'pfx_z': phys.get('pfx_z', 0),
            'sz_top': phys.get('sz_top', 3.5),
            'sz_bot': phys.get('sz_bot', 1.5),
            
            # Derived Physics
            'remaining_frames': remaining_frames,
            'phys_pred_x': phys_pred_x,
            'phys_pred_z': phys_pred_z,
            
            # Context
            'is_right_hand_batter': 1 if phys.get('stand') == 'R' else 0,
            'is_right_hand_pitcher': 1 if phys.get('p_throws') == 'R' else 0
        }
        
        features.append(row_feat)
        targets_x.append(data['labels']['plate_x'])
        targets_z.append(data['labels']['plate_z'])
        meta_labels.append(data['labels']) 

    return pd.DataFrame(features), np.array(targets_x), np.array(targets_z), meta_labels

# 2. Logic Engines
def determine_pitch_result(pred_x, pred_z, sz_top, sz_bot):
    """
    Determines Strike vs Ball using the formula:
    -(17/24 + r) <= plate_x <= (17/24 + r)
    """
    # 17/24 is exactly half the plate width in feet
    half_width = PLATE_HALF_WIDTH_FT 
    r = BALL_RADIUS_FT
    
    width_limit = half_width + r
    
    in_width = -width_limit <= pred_x <= width_limit
    in_height = (sz_bot - r) <= pred_z <= (sz_top + r)
    
    return "strike" if (in_width and in_height) else "ball"

def determine_zone(pred_x, pred_z, sz_top, sz_bot, pitch_class):
    """
    Determines the zone (1-14).
    """
    
    # --- IF BALL ---
    if pitch_class == 'ball':
        # Split by center point
        center_x = 0
        center_z = (sz_top + sz_bot) / 2
        
        is_upper = pred_z >= center_z
        is_left = pred_x < center_x
        
        if is_upper and is_left:
            return 11
        elif is_upper and not is_left:
            return 12
        elif not is_upper and is_left:
            return 13
        else: # not upper and not left
            return 14

    # --- IF STRIKE ---
    # Zone 1-9 is defined by the actual plate width (17 inches), NOT the buffer.
    # We strictly cut the plate into 3 equal columns.
    
    width_third = PLATE_WIDTH_FT / 3 
    left_cut = -width_third / 2
    right_cut = width_third / 2
    
    col = 0
    if pred_x < left_cut:
        col = 0 # Left Side (Visual Left) -> Corresponds to [3, 6, 9]
    elif pred_x > right_cut:
        col = 2 # Right Side (Visual Right) -> Corresponds to [1, 4, 7]
    else:
        col = 1 # Center -> Corresponds to [2, 5, 8]
        
    # Vertical cuts
    height = sz_top - sz_bot
    height_third = height / 3
    top_cut = sz_top - height_third
    bot_cut = sz_bot + height_third
    
    row = 0
    if pred_z > top_cut:
        row = 0 # Top Row
    elif pred_z < bot_cut:
        row = 2 # Bottom Row
    else:
        row = 1 # Middle Row
        
    # Map (Row, Col) to Zone Number
    # Grid:
    #      Col0  Col1  Col2
    # Row0  3     2     1
    # Row1  6     5     4
    # Row2  9     8     7
    
    zone_map = [
        [1, 2, 3], # Top Row
        [4, 5, 6], # Middle Row
        [7, 8, 9]  # Bottom Row
    ]
    
    return zone_map[row][col]

# --- MAIN EXECUTION ---
if __name__ == "__main__":

    train_dataset, val_dataset = create_data_loaders(CSV_PATH, VIDEO_DIR, YOLO_PATH)
    
    # Extract Features (TRAIN)
    X_train, y_x_train, y_z_train, _ = extract_features(train_dataset, desc="Processing Train Data")
    
    # Extract Features (VALIDATION)
    X_val, y_x_val, y_z_val, metas_val = extract_features(val_dataset, desc="Processing Validation Data")
    
    # Train Models
    print("\nTraining XGBoost Models on Train Set...")
    model_x = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, max_depth=5, learning_rate=0.05)
    model_z = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, max_depth=5, learning_rate=0.05)
    
    model_x.fit(X_train, y_x_train)
    model_z.fit(X_train, y_z_train)
    
    # Evaluate on Validation Set
    print("\nEvaluating on Untested Validation Set...")
    preds_x = model_x.predict(X_val)
    preds_z = model_z.predict(X_val)
    
    # Metrics
    mae_x = mean_absolute_error(y_x_val, preds_x)
    mae_z = mean_absolute_error(y_z_val, preds_z)
    print(f"Validation MAE (X): {mae_x:.4f} ft")
    print(f"Validation MAE (Z): {mae_z:.4f} ft")
    
    correct_class = 0
    total = 0
    validation_results = []
    
    for i in range(len(preds_x)):
        p_x = preds_x[i]
        p_z = preds_z[i]
        
        # Get ground truth context from validation meta labels
        sz_top = X_val.iloc[i]['sz_top']
        sz_bot = X_val.iloc[i]['sz_bot']
        
        # Predict Class
        pred_class = determine_pitch_result(p_x, p_z, sz_top, sz_bot)
        pred_zone = determine_zone(p_x, p_z, sz_top, sz_bot, pred_class)
        
        # Compare to Truth
        true_class = metas_val[i]['pitch_class']
        if pred_class == true_class:
            correct_class += 1
        total += 1
        
        # Store for analysis
        validation_results.append({
            'true_x': y_x_val[i],
            'pred_x': p_x,
            'true_z': y_z_val[i],
            'pred_z': p_z,
            'true_class': true_class,
            'pred_class': pred_class,
            'true_zone': metas_val[i]['zone'],
            'pred_zone': pred_zone
        })
        
    print(f"Validation Pitch Class Accuracy: {correct_class/total:.2%}")
    
    # Save Validation Analysis
    pd.DataFrame(validation_results).to_csv("validation_analysis.csv", index=False)
    print("Saved validation analysis to 'validation_analysis.csv'")


    # ==========================================
    # 8. TEST SET PREDICTION
    # ==========================================
    print("\n--- Starting Test Set Inference ---")
    
    if Path(TEST_CSV_PATH).exists():
        test_df = pd.read_csv(TEST_CSV_PATH)
        print(f"Loaded Test Data: {len(test_df)} samples from {TEST_CSV_PATH}")
        
        test_dataset = PitchDataset(test_df, TEST_VIDEO_DIR, YOLO_PATH)
        
        # Extract features (targets will be 0/dummy, which we ignore)
        X_test, _, _, _ = extract_features(test_dataset, desc="Processing Test Data")
        
        # Predict
        print("Predicting on Test Set...")
        test_preds_x = model_x.predict(X_test)
        test_preds_z = model_z.predict(X_test)
        
        submission_rows = []
        
        for i in range(len(test_preds_x)):
            p_x = test_preds_x[i]
            p_z = test_preds_z[i]
            
            # Get context from features (sz_top/sz_bot are in X_test)
            sz_top = X_test.iloc[i]['sz_top']
            sz_bot = X_test.iloc[i]['sz_bot']
            
            # Logic
            pred_class = determine_pitch_result(p_x, p_z, sz_top, sz_bot)
            pred_zone = determine_zone(p_x, p_z, sz_top, sz_bot, pred_class)
            
            # Use original filename from dataframe
            fname = test_df.iloc[i]['file_name']
            
            submission_rows.append({
                'file_name': fname,
                'pitch_class': pred_class,
                'zone': pred_zone
            })
            
        # Save Submission
        sub_df = pd.DataFrame(submission_rows)
        sub_df.to_csv("submission.csv", index=False)
        print(f"Saved {len(sub_df)} predictions to 'submission.csv'")
        print(sub_df.head())
        
    else:
        print(f"Test CSV not found at {TEST_CSV_PATH}. Skipping inference.")