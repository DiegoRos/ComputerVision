import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm
import sys
import os

from pitch_dataset import PitchDataset
from pitch_model import MultiTaskPitchModel


PHYSICS_COLS = [
    'release_speed', 'release_spin_rate', 'release_extension',
    'release_pos_x','release_pos_y', 'release_pos_z', 'pfx_x', 'pfx_z',
    'sz_top', 'sz_bot', 'stand', 'p_throws'
]

OUTPUT_CSV = "submission.csv"
ROOT = Path("/content/drive/MyDrive/Fall 2025/CS-GY 6643 Computer Vision/Project 3/competition_folder/baseball-pitch-tracking/kaggle_dataset")
TEST_VIDEO_DIR = ROOT / "test"

data_path = ROOT / "data"
TEST_CSV_PATH = data_path / "test_features.csv"

YOLO_PATH = "/content/drive/MyDrive/Fall 2025/CS-GY 6643 Computer Vision/Project 3/competition_folder/fine_tuning/baseball_rubber_home_glove/model_output/ball_finetune/weights/best.pt"

MODEL_WEIGHTS_PATH = "/content/drive/MyDrive/Fall 2025/CS-GY 6643 Computer Vision/Project 3/competition_folder/multitask_model.pth"

# PURE AI CODE WRITTEN AT 3 AM CHECK TOMORROW
def get_zone_and_class(plate_x, plate_z, sz_top, sz_bot, ball_radius_ft=0.12):
    """
    Determines the pitch class (Strike/Ball) and the specific Gameday Zone (1-14).

    Args:
        plate_x (float): Horizontal location of the ball at the plate (feet).
                         Negative = Left (Catcher's View), Positive = Right.
        plate_z (float): Vertical location of the ball at the plate (feet).
        sz_top (float): Top of the batter's strike zone (feet).
        sz_bot (float): Bottom of the batter's strike zone (feet).
        ball_radius_ft (float): Radius of the ball to determine 'touching' the zone.
                                Standard is ~1.45 inches -> 0.12 feet.

    Returns:
        pitch_class (str): "strike" or "ball"
        zone (int): 1-14
    """

    # --- 1. Define Boundaries ---
    # The plate is 17 inches wide. Half-width is 8.5 inches.
    # 8.5 inches / 12 = 0.7083 feet.
    half_plate = 17 / 24

    # Strike definition: The BALL (with radius) must touch the ZONE volume.
    # So we expand the zone limits by the ball's radius.
    # Note: Some competitions strictly use the center of the ball.
    # If strictly center, set ball_radius_ft = 0.

    left_boundary = -(half_plate + ball_radius_ft)
    right_boundary = (half_plate + ball_radius_ft)
    bot_boundary = sz_bot - ball_radius_ft
    top_boundary = sz_top + ball_radius_ft

    # --- 2. Determine Strike vs Ball ---
    is_strike = (left_boundary <= plate_x <= right_boundary) and \
                (bot_boundary <= plate_z <= top_boundary)

    pitch_class = "strike" if is_strike else "ball"

    # --- 3. Determine Zone (1-9) ---
    # Zones 1-9 represent the 3x3 grid inside the strike zone.
    # We define the internal grid lines based on the ACTUAL plate width (no radius padding for grid).
    # This ensures the visual grid aligns with the physical plate.

    # Grid width = 17 inches / 3
    grid_w = (17/12) / 3

    # Grid height = (Top - Bot) / 3
    grid_h = (sz_top - sz_bot) / 3

    if is_strike:
        # Columns: 0 = Left, 1 = Middle, 2 = Right (Catcher's Perspective)
        # Remember: plate_x < -grid_w/2 is LEFT
        if plate_x < -grid_w/2:
            col = 0 # Left
        elif plate_x > grid_w/2:
            col = 2 # Right
        else:
            col = 1 # Middle

        # Rows: 0 = Top, 1 = Middle, 2 = Bottom
        if plate_z > (sz_top - grid_h):
            row = 0 # Top
        elif plate_z < (sz_bot + grid_h):
            row = 2 # Bottom
        else:
            row = 1 # Middle

        # Formula to map (row, col) to Keypad (1-9)
        # Row 0: 0,1,2 -> 1,2,3
        # Row 1: 0,1,2 -> 4,5,6
        # Row 2: 0,1,2 -> 7,8,9
        zone = (row * 3) + col + 1

    else:
        # --- 4. Determine Shadow Zones (11-14) ---
        # The prompt image defines 4 outer quadrants.
        # 11: Top-Left (High & Left)
        # 12: Top-Right (High & Right)
        # 13: Bot-Left (Low & Left)
        # 14: Bot-Right (Low & Right)

        # We use the center of the zone to split Left/Right and Top/Bottom
        # Left/Right Split is x=0
        # Top/Bottom Split is midpoint of vertical zone

        mid_z = (sz_top + sz_bot) / 2

        is_left = plate_x < 0
        is_high = plate_z > mid_z

        if is_high and is_left:
            zone = 11
        elif is_high and not is_left:
            zone = 12
        elif not is_high and is_left:
            zone = 13
        else: # Low and Right
            zone = 14

    return pitch_class, zone

def prepare_test_dataframe(csv_path):
    """
    Loads test features and adds dummy target columns so PitchDataset works.
    Also performs the normalization step.
    """
    df = pd.read_csv(csv_path)

    # 1. Add Dummy Targets if missing (Test data won't have answers)
    if 'plate_x' not in df.columns: df['plate_x'] = 0.0
    if 'plate_z' not in df.columns: df['plate_z'] = 0.0
    if 'pitch_class' not in df.columns: df['pitch_class'] = 'ball'
    if 'zone' not in df.columns: df['zone'] = 0

    # 2. Normalize Physics Columns
    # NOTE: Ideally, you use the Mean/Std from the TRAINING set here.
    # Since we don't have those loaded, we normalize based on Test distribution
    # (standard practice in simple pipelines, though slightly transductive).
    numeric_cols = [c for c in PHYSICS_COLS if c not in ['stand', 'p_throws']]

    for col in numeric_cols:
        std_val = df[col].std()
        if pd.isna(std_val) or std_val == 0:
            std_val = 1.0

        # Create the _std column PitchDataset looks for
        df[f"{col}_std"] = (df[col] - df[col].mean()) / std_val
        df[f"{col}_std"] = df[f"{col}_std"].fillna(0.0)

    return df



HALF_PLATE_FT = 17/24.0  # 0.708333...

def denorm_to_feet(x_lbl, z_lbl, sz_top, sz_bot):
    mid = 0.5*(sz_top + sz_bot)
    half_h = 0.5*(sz_top - sz_bot)
    x_ft = x_lbl * HALF_PLATE_FT
    z_ft = mid + z_lbl * half_h
    return float(x_ft), float(z_ft)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # --------- DATA ----------
    print(f"Loading test data from {TEST_CSV_PATH}...")
    test_df = prepare_test_dataframe(TEST_CSV_PATH)

    # Initialize Dataset
    # We use mode='test' just for clarity, though logic is same
    test_ds = PitchDataset(test_df, str(TEST_VIDEO_DIR), YOLO_PATH, mode='test')

    # IMPORTANT: batch_size=1 and shuffle=False to easily match filenames
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    # --------- MODEL ----------
    model = MultiTaskPitchModel(physics_dim=9).to(device)
    if os.path.exists(MODEL_WEIGHTS_PATH):
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    else:
        print(f"WARNING: No weights at {MODEL_WEIGHTS_PATH} (using random).")
    model.eval()

    use_calib = True  # set False if you didn't fit on val previously
    # (Optional) fit calibration on the same split
    @torch.no_grad()
    def fit_calibration(model, loader, device, max_batches=40):
        px=[]; pz=[]; gx=[]; gz=[]
        for i, batch in enumerate(loader):
            if i>=max_batches: break
            traj = batch['trajectory'].to(device)
            phys = batch['physics'].to(device)
            y = batch['labels'][:, :2].cpu()
            pc,_ = model(traj, phys)
            px.append(pc[:,0].cpu()); pz.append(pc[:,1].cpu())
            gx.append(y[:,0]);       gz.append(y[:,1])
        px=torch.cat(px); pz=torch.cat(pz); gx=torch.cat(gx); gz=torch.cat(gz)
        A = torch.stack([px, torch.ones_like(px)], 1)
        ax,bx = torch.linalg.lstsq(A, gx).solution
        A = torch.stack([pz, torch.ones_like(pz)], 1)
        az,bz = torch.linalg.lstsq(A, gz).solution
        print(f"[calib] x_lbl ≈ {ax:.4f}*pred + {bx:.4f}")
        print(f"[calib] z_lbl ≈ {az:.4f}*pred + {bz:.4f}")
        return float(ax), float(bx), float(az), float(bz)

    use_calib = True
    if use_calib:
        ax,bx,az,bz = fit_calibration(model, test_loader, device)


    def apply_calibration(px_lbl, pz_lbl):
        if use_calib:
            px_lbl = ax * px_lbl + bx
            pz_lbl = az * pz_lbl + bz
        return float(px_lbl), float(pz_lbl)

    def pred_to_feet(px_lbl, pz_lbl, sz_top, sz_bot):
        px_lbl, pz_lbl = apply_calibration(px_lbl, pz_lbl)
        return denorm_to_feet(px_lbl, pz_lbl, sz_top, sz_bot)

    # --------- INFERENCE (TEST: no ground truth) ----------
    rows = []
    with torch.no_grad():
        for batch, (_, row) in tqdm(zip(test_loader, test_df.iterrows()),
                                    total=len(test_loader)):
            # Inputs
            traj = batch['trajectory'].to(device)
            phys = batch['physics'].to(device)

            # Per-pitch strike zone bounds
            phys_raw = batch['physics_raw'][0].cpu()
            sz_top = float(phys_raw[7])
            sz_bot = float(phys_raw[8])

            # Predict normalized coords and class logit
            pred_coords, pred_logits = model(traj, phys)
            px_lbl = float(pred_coords[0, 0].cpu())
            pz_lbl = float(pred_coords[0, 1].cpu())

            # Apply calibration -> feet
            pred_x_ft, pred_z_ft = pred_to_feet(px_lbl, pz_lbl, sz_top, sz_bot)

            # Zone & strike/ball (prediction only)
            _, pred_zone = get_zone_and_class(pred_x_ft, pred_z_ft, sz_top, sz_bot)

            prob = torch.sigmoid(pred_logits.squeeze(0)).item()  # robust squeeze
            pred_cls = 1 if prob >= 0.5 else 0

            # Filename from the dataframe
            fname = row['file_name']

            pred_cls_label = "ball" if pred_cls == 0 else "strike"

            # Store only predictions (no GT on test)
            rows.append({
                "file_name": fname,
                "pitch_class": pred_cls_label,  # 1=strike, 0=ball
                "zone": int(pred_zone)
            })

    # --------- SAVE SUBMISSION ----------
    submission_df = pd.DataFrame(rows)
    submission_df.to_csv(OUTPUT_CSV, index=False)
    print("Saved:", OUTPUT_CSV)
    print(submission_df.head())


if __name__ == "__main__":
    main()