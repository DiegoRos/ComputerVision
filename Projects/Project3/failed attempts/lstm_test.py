import torch
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

# Use the imports as defined in lstm_train.py
from lstm_dataset import PitchDataset
from lstm_model import HybridPitchModel

# --- Configuration ---
BATCH_SIZE = 1 # Inference is safer with batch 1 for mapping back to filenames easily
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
ROOT_PATH = Path.cwd()
COMPETITION_ROOT = ROOT_PATH / "competition_folder/baseball-pitch-tracking/kaggle_dataset"

TEST_VIDEO_DIR = COMPETITION_ROOT / "test"
TEST_CSV_PATH = COMPETITION_ROOT / "data/test_features.csv"

# Model & Stats Paths
# Assumes stats.json is in a folder named 'norm_stats' in the current working directory as per previous step
STATS_PATH = ROOT_PATH / "norm_stats/stats.json" 

YOLO_PATH = "./competition_folder/model_output/ball_finetune/weights/best.pt"
MODEL_NAME = "lstm_multitask_model__5epochs.pth"
# Using the specific path you requested
MODEL_WEIGHTS_PATH = f"competition_folder/{MODEL_NAME}"

OUTPUT_DIR = ROOT_PATH / "submission"
OUTPUT_CSV = OUTPUT_DIR / "submission.csv"

def get_pitch_class(plate_x, plate_z, sz_top, sz_bot):
    """
    Determines Strike vs Ball based on MLB Rule.
    Strike zone width is 17 inches (17/12 feet).
    Ball radius is approx 1.45 inches (1.45/12 feet).
    """
    # Constants in feet
    ball_radius = 1.45 / 12
    plate_width_half = (17 / 12) / 2
    
    # Horizontal limit (left/right of center)
    x_limit = plate_width_half + ball_radius
    
    # Vertical limit
    z_top_limit = sz_top + ball_radius
    z_bot_limit = sz_bot - ball_radius
    
    is_strike_width = -x_limit <= plate_x <= x_limit
    is_strike_height = z_bot_limit <= plate_z <= z_top_limit
    
    if is_strike_width and is_strike_height:
        return "strike"
    else:
        return "ball"

def get_zone(plate_x, plate_z, sz_top, sz_bot):
    """
    Maps plate coordinates to the 1-9 grid or 11-14 shadow zones.
    Standard Catcher's View:
    x < 0 is Left (Third Base side), x > 0 is Right (First Base side).
    """
    # Define the core strike zone boundaries (without ball radius buffer for the grid itself)
    width_per_third = (17 / 12) / 3
    height = sz_top - sz_bot
    height_per_third = height / 3
    
    # Check if inside the 3x3 grid (Core Strike Zone)
    # Using strict definition for 1-9, everything else is 11-14
    half_plate = (17/12) / 2
    
    if (-half_plate <= plate_x <= half_plate) and (sz_bot <= plate_z <= sz_top):
        # It is in 1-9
        
        # Determine Column (1, 2, 3) -> (Left, Middle, Right)
        if plate_x < -width_per_third/2:
            col = 0 # Left (Zones 1, 4, 7)
        elif plate_x > width_per_third/2:
            col = 2 # Right (Zones 3, 6, 9)
        else:
            col = 1 # Middle (Zones 2, 5, 8)
            
        # Determine Row (0, 1, 2) -> (Top, Middle, Bottom)
        # Note: z increases upwards. 
        # Top third: > sz_top - height/3
        if plate_z > (sz_top - height_per_third):
            row = 0 # Top (Zones 1, 2, 3)
        elif plate_z < (sz_bot + height_per_third):
            row = 2 # Bottom (Zones 7, 8, 9)
        else:
            row = 1 # Middle (Zones 4, 5, 6)
            
        # Map (row, col) to Zone Number (from pitchers perspective)
        # Row 0: 3, 2, 1
        # Row 1: 6, 5, 4
        # Row 2: 9, 8, 7
        zone_map = [
            [3, 2, 1],
            [6, 5, 4],
            [9, 8, 7]
        ]
        return zone_map[row][col]
        
    else:
        # It is in 11-14 (Shadow/Waste)
        # 12: Top-Left, 11: Top-Right, 14: Bot-Left, 13: Bot-Right
        # We split based on the center of the zone
        
        is_left = plate_x <= 0
        is_top = plate_z >= (sz_bot + height/2)
        
        if is_left and is_top:
            return 12
        elif not is_left and is_top:
            return 11
        elif is_left and not is_top:
            return 14
        else:
            return 13

def main():
    print("--- Starting Inference Pipeline ---")
    
    # 1. Create Output Directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 2. Load Stats
    print(f"Loading normalization stats from {STATS_PATH}...")
    if not STATS_PATH.exists():
        raise FileNotFoundError(f"Stats file not found at {STATS_PATH}. Did you run training?")
    
    with open(STATS_PATH, 'r') as f:
        stats = json.load(f)
        
    # 3. Load Data
    print(f"Loading Test Metadata from {TEST_CSV_PATH}...")
    df_test = pd.read_csv(TEST_CSV_PATH)
    
    # Initialize Dataset in TEST mode with loaded stats
    test_dataset = PitchDataset(
        data_frame=df_test,
        video_root_dir=str(TEST_VIDEO_DIR),
        yolo_model_path=YOLO_PATH,
        mode='test', # Important: returns simplified dictionary (no labels)
        stats=stats
    )
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 4. Load Model
    print(f"Loading Model from {MODEL_WEIGHTS_PATH}...")
    model = HybridPitchModel(physics_dim=13).to(DEVICE)
    
    if torch.cuda.is_available():
        checkpoint = torch.load(MODEL_WEIGHTS_PATH)
    else:
        checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location=torch.device('cpu'))
        
    model.load_state_dict(checkpoint)
    model.eval()
    
    # 5. Inference Loop
    results = []
    
    print("Running predictions...")
    with torch.no_grad():
        # Iterate over loader
        # We need to map predictions back to filenames. 
        # Since shuffle=False and batch_size=1 (or we track indices), we can align with df_test.
        
        for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            trajectory = batch['trajectory'].to(DEVICE)
            physics = batch['physics'].to(DEVICE)
            
            # Forward Pass
            preds_norm = model(trajectory, physics)
            
            # Denormalize to get Feet
            preds_feet = test_dataset.denormalize_targets(preds_norm)
            
            # Since batch_size might be > 1, we iterate through results in this batch
            for j in range(len(preds_feet)):
                global_idx = i * BATCH_SIZE + j
                if global_idx >= len(df_test): break
                
                row_data = df_test.iloc[global_idx]
                file_name = row_data['file_name']
                
                # Get predicted values
                pred_x = preds_feet[j][0]
                pred_z = preds_feet[j][1]
                
                # Get Strike Zone limits for this specific batter
                sz_top = row_data['sz_top']
                sz_bot = row_data['sz_bot']
                
                # Logic Rules
                pitch_class = get_pitch_class(pred_x, pred_z, sz_top, sz_bot)
                zone = get_zone(pred_x, pred_z, sz_top, sz_bot)
                
                results.append({
                    'file_name': file_name,
                    'pitch_class': pitch_class,
                    'zone': int(zone)
                })

    # 6. Save Submission
    submission_df = pd.DataFrame(results)
    print(f"Saving results to {OUTPUT_CSV}...")
    submission_df.to_csv(OUTPUT_CSV, index=False)
    print("Done!")
    print(submission_df.head())

if __name__ == "__main__":
    main()