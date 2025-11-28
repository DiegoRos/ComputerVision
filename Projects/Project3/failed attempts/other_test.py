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

from other_pitch_dataset import PitchDataset
from other_pitch_model import MultiTaskPitchModel


PHYSICS_COLS = [
    'release_speed', 'release_spin_rate', 'release_extension',
    'release_pos_x','release_pos_y', 'release_pos_z', 'pfx_x', 'pfx_z',
    'sz_top', 'sz_bot', 'stand', 'p_throws'
]

OUTPUT_PATH = Path.cwd() / "submission"

OUTPUT_CSV = "submission.csv"
root_path = Path.cwd()
ROOT = root_path / "competition_folder/baseball-pitch-tracking/kaggle_dataset"

TEST_VIDEO_DIR = ROOT / "test"

data_path = ROOT / "data"
TEST_CSV_PATH = data_path / "test_features.csv"

YOLO_PATH = "./competition_folder/model_output/ball_finetune/weights/best.pt"

MODEL_NAME= "multitask_model_20epochs.pth"
MODEL_WEIGHTS_PATH = f"competition_folder/{MODEL_NAME}"

zone_map = {
    1:0, 2:1, 3:2, 
    4:3, 5:4, 6:5, 
    7:6, 8:7, 9:8, 
    11:9, 12:10, 13:11, 14:12
}

map_zone = {v:k for k,v in zone_map.items()}

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
    print("Loading model...")
    model = MultiTaskPitchModel(physics_dim=9).to(device)
    if os.path.exists(MODEL_WEIGHTS_PATH):
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    else:
        print(f"WARNING: No weights at {MODEL_WEIGHTS_PATH} (using random).")
    model.eval()

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
            pred_coords, pred_logits, pred_zone_logits = model(traj, phys)


            prob = torch.sigmoid(pred_logits.squeeze(0)).item()  # robust squeeze
            pred_cls = 1 if prob >= 0.5 else 0

            pred_zone = torch.argmax(pred_zone_logits, dim=1).item()

            # Filename from the dataframe
            fname = row['file_name']

            pred_cls_label = "ball" if pred_cls == 0 else "strike"
            pred_zone_label = map_zone.get(pred_zone, 5 if pred_cls == 0 else 14)  # Default to 14 if not found

            # Store only predictions (no GT on test)
            rows.append({
                "file_name": fname,
                "pitch_class": pred_cls_label,  # 1=strike, 0=ball
                "zone": int(pred_zone_label)
            })

    # --------- SAVE SUBMISSION ----------
    submission_df = pd.DataFrame(rows)
    submission_df.to_csv(OUTPUT_PATH / f"{MODEL_NAME}_{OUTPUT_CSV}", index=False)
    print("Saved:", OUTPUT_CSV)
    print(submission_df.head())


if __name__ == "__main__":
    main()