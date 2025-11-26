import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import pandas as pd
import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Import your custom modules
from lstm_dataset import PitchDataset
from lstm_model import HybridPitchModel

# --- Configuration ---
BATCH_SIZE = 4
GRAD_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5

root_path = Path.cwd()
competition_path = root_path / "competition_folder/baseball-pitch-tracking/kaggle_dataset"
data_path = competition_path / "data"

train_video_dir = competition_path / "train_trimmed"
train_csv_path = data_path / "train_ground_truth.csv"

VIDEO_ROOT = str(train_video_dir)
CSV_PATH = str(train_csv_path)
YOLO_PATH = "./competition_folder/model_output/ball_finetune/weights/best.pt"


# For training the model we want to create a train/val/test split with a 80/20 spilt
def create_data_loaders(csv_path, video_dir, yolo_model_dir, batch_size=4, shuffle=True):
    # Load csv
    df = pd.read_csv(str(csv_path))
    df = df.iloc[0:100] # TEMP: For quick testing, remove this line for full data

    # Create first split (80/20) spliting into train and val/test
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42, # For reproducibility
        stratify=df['pitch_class'] # Ensure that we get a smiliar distribution of ball/strike in testing set
    )


    print(f"Full Dataset: {len(df)}")
    print(f"Train Set:    {len(train_df)} ({(len(train_df)/len(df))*100:.1f}%)")
    print(f"Val Set:      {len(val_df)} ({(len(val_df)/len(df))*100:.1f}%)")

    # Create Torch datasets
    train_ds = PitchDataset(train_df, video_dir, yolo_model_dir, mode='train')
    
    # Get stats from training set to prevent leakage
    train_stats = train_ds.get_stats()
    
    # Pass stats to validation set
    val_ds = PitchDataset(val_df, video_dir, yolo_model_dir, mode='train', stats=train_stats)

    # Create Data Loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, train_ds, val_loader, val_ds

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
        return 1
    else:
        return 0

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

def denormalize_value(val, mean, std):
    return (val * std) + mean

def validate(model, val_loader, criterion, device, dataset_obj):
    """
    Runs validation loop and calculates:
    1. MSE Loss
    2. MAE in Feet
    3. Confusion Matrix (Strike/Ball)
    4. Zone Accuracy
    """
    model.eval()
    total_loss = 0.0
    total_mae_feet = 0.0
    num_batches = 0
    
    # Storage for classification metrics
    all_true_classes = []
    all_pred_classes = []
    all_true_zones = []
    all_pred_zones = []

    # Get means/stds for denormalizing physics (sz_top/sz_bot)
    stats = dataset_obj.get_stats()
    mean_sz_top = stats['means']['sz_top']
    std_sz_top = stats['stds']['sz_top']
    mean_sz_bot = stats['means']['sz_bot']
    std_sz_bot = stats['stds']['sz_bot']

    with torch.no_grad():
        pbar = tqdm(val_loader,total=len(val_loader), desc="Validating", leave=False)
        for batch in pbar:
            # Move data to device
            trajectory = batch['trajectory'].to(device)
            physics = batch['physics'].to(device)
            targets = batch['labels'].to(device)
            
            # Ground Truth Classification Labels
            true_class_labels = batch['class_label'].cpu().numpy() # 1.0 or 0.0
            true_zone_labels = batch['zone_label'].cpu().numpy()   # 1-14
            
            # Forward Pass
            outputs = model(trajectory, physics)
            
            # Loss & MAE
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            preds_feet = dataset_obj.denormalize_targets(outputs)
            targets_feet = dataset_obj.denormalize_targets(targets)
            
            mae = np.mean(np.abs(preds_feet - targets_feet))
            total_mae_feet += mae
            
            # --- Logic Rule Evaluation ---
            # We need sz_top and sz_bot to determine the zone.
            # In dataset.py, 'physics' tensor indices:
            # 7: sz_top, 8: sz_bot (Normalized)
            sz_top_norm = physics[:, 7].cpu().numpy()
            sz_bot_norm = physics[:, 8].cpu().numpy()
            
            # Denormalize to get real feet for logic check
            sz_top_feet = denormalize_value(sz_top_norm, mean_sz_top, std_sz_top)
            sz_bot_feet = denormalize_value(sz_bot_norm, mean_sz_bot, std_sz_bot)
            
            # Loop through batch to apply logic rules
            for i in range(len(preds_feet)):
                px = preds_feet[i][0]
                pz = preds_feet[i][1]
                top = sz_top_feet[i]
                bot = sz_bot_feet[i]
                
                # Predict Strike/Ball
                p_class = get_pitch_class(px, pz, top, bot)
                all_pred_classes.append(p_class)
                all_true_classes.append(int(true_class_labels[i]))
                
                # Predict Zone
                p_zone = get_zone(px, pz, top, bot)
                all_pred_zones.append(p_zone)
                all_true_zones.append(int(true_zone_labels[i]))
            
            num_batches += 1
            
    avg_loss = total_loss / num_batches
    avg_mae = total_mae_feet / num_batches
    
    # --- Print Metrics ---
    print("\n  >>> Classification Metrics:")
    
    # Confusion Matrix
    cm = confusion_matrix(all_true_classes, all_pred_classes, labels=[0, 1])
    print(f"  Confusion Matrix (Ball=0, Strike=1):\n{cm}")
    if len(all_true_classes) > 0:
        acc = np.sum(np.array(all_true_classes) == np.array(all_pred_classes)) / len(all_true_classes)
        print(f"  Strike/Ball Accuracy: {acc*100:.2f}%")
        
    # Zone Accuracy
    true_zones_np = np.array(all_true_zones)
    pred_zones_np = np.array(all_pred_zones)
    
    print("  Zone Accuracy Breakdown:")
    # Check accuracy for each unique zone present in validation
    unique_zones = np.unique(true_zones_np)
    for z in sorted(unique_zones):
        mask = true_zones_np == z
        if np.sum(mask) > 0:
            z_acc = np.sum(pred_zones_np[mask] == true_zones_np[mask]) / np.sum(mask)
            print(f"    Zone {z}: {z_acc*100:.1f}% ({np.sum(mask)} samples)")
            
    return avg_loss, avg_mae


def main():
    # 1. Device Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # 2. Prepare Data
    if os.path.exists(CSV_PATH):
        train_loader, train_dataset, val_loader, val_dataset = create_data_loaders(CSV_PATH, VIDEO_ROOT, YOLO_PATH, batch_size=BATCH_SIZE)
        
        # Save Normalization Stats
        root_path = Path.cwd()
        stats_dir = root_path / "norm_stats"
        stats_dir.mkdir(exist_ok=True)
        stats_path = stats_dir / "stats.json"
        print(f"Saving normalization stats to {stats_path}...")
        train_dataset.save_stats(str(stats_path))
        
    else:
        print(f"Error: {CSV_PATH} not found. Please provide actual data path.")
        return

    # 3. Model Setup
    model = HybridPitchModel(physics_dim=13).to(device)
    
    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss() # Standard regression loss

    # Initialize best validation loss for saving best model
    best_val_loss = float('inf')

    # 4. Training Loop
    print("Starting training...")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move inputs to device
            trajectory = batch['trajectory'].to(device)
            physics = batch['physics'].to(device)
            targets = batch['labels'].to(device)
            
            # Forward Pass
            outputs = model(trajectory, physics)
            
            # Loss Calculation
            loss = criterion(outputs, targets)
            
            # Normalize loss
            loss = loss / GRAD_ACCUMULATION_STEPS
            
            # Backward Pass
            loss.backward()
            
            # Gradient Accumulation
            if (batch_idx + 1) % GRAD_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * GRAD_ACCUMULATION_STEPS
            progress_bar.set_postfix(loss=loss.item() * GRAD_ACCUMULATION_STEPS)
            
        if len(train_loader) % GRAD_ACCUMULATION_STEPS != 0:
            optimizer.step()
            optimizer.zero_grad()

        # Validation Step
        avg_train_loss = epoch_loss / len(train_loader)
        val_loss, val_mae = validate(model, val_loader, criterion, device, val_dataset)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train MSE (Norm): {avg_train_loss:.4f}")
        print(f"  Val MSE (Norm)  : {val_loss:.4f}")
        print(f"  Val MAE (Real)  : {val_mae:.4f} feet")
        
        # Save Best Model Logic
        if val_loss < best_val_loss:
            print(f"  >>> Best Model Found! (Val Loss: {val_loss:.4f} < {best_val_loss:.4f}) Saving...")
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"competition_folder/best_hybrid_pitch_model_{NUM_EPOCHS}epochs.pth")
        else:
            print(f"  >>> Val Loss did not improve (Best: {best_val_loss:.4f}).")

    print("Training Complete.")

if __name__ == "__main__":
    main()