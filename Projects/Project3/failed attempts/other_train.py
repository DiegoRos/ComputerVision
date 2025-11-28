import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
from tqdm import tqdm
import sys

from other_pitch_dataset import PitchDataset
from other_pitch_model import MultiTaskPitchModel

PHYSICS_COLS = [
    'release_speed', 'release_spin_rate', 'release_extension',
    'release_pos_x','release_pos_y', 'release_pos_z', 'pfx_x', 'pfx_z',
    'sz_top', 'sz_bot', 'stand', 'p_throws'
]

PHYSICS_COLS = [
    'release_speed', 'release_spin_rate', 'release_extension',
    'release_pos_x','release_pos_y', 'release_pos_z', 'pfx_x', 'pfx_z',
    'sz_top', 'sz_bot', 'stand', 'p_throws'
]
# effective_speed is a function of release_speed and the release position.
# This means that adding effective_speed will only add multicolinearity to the NN inputs
# This can add additional noise and affect convergence, so it will not be used for now (we can test if it improves performance later)

TARGET_COLS = ['plate_x', 'plate_z', 'pitch_class']

csv_train_header_labels = [
    "file_name", "plate_x", "plate_z", "sz_top", "sz_bot", "release_speed",
    "effective_speed", "release_spin_rate", "release_pos_x", "release_pos_y",
    "release_pos_z", "release_extension", "pfx_x", "pfx_z", "stand",
    "p_throws", "pitch_class", "zone"
]

csv_test_header_labels = [
    "file_name", "sz_top", "sz_bot", "release_speed", "effective_speed",
    "release_spin_rate", "release_pos_x", "release_pos_y", "release_pos_z",
    "release_extension", "pfx_x", "pfx_z", "stand", "p_throws"
]

# For training the model we want to create a train/val/test split with a 70/20/10 spilt
def create_data_loaders(csv_path, video_dir, yolo_model_dir, batch_size=4, shuffle=True):
    # Load csv
    df = pd.read_csv(str(csv_path))

    # Create a normalized version of the numerical values of the df appended to the end, save them as <col>_std
    numeric_cols = [c for c in PHYSICS_COLS if c not in ['stand', 'p_throws']]
    for col in numeric_cols:
        df[f"{col}_std"] = (df[col] - df[col].mean()) / df[col].std()


    # Create first split (70/30) spliting into train and val/test
    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        random_state=42, # For reproducibility
        stratify=df['pitch_class'] # Ensure that we get a smiliar distribution of ball/strike in testing set
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=1/3,
        random_state=42,
        stratify=temp_df['pitch_class']
    )

    print(f"Full Dataset: {len(df)}")
    print(f"Train Set:    {len(train_df)} ({(len(train_df)/len(df))*100:.1f}%)")
    print(f"Val Set:      {len(val_df)} ({(len(val_df)/len(df))*100:.1f}%)")
    print(f"Test Set:     {len(test_df)} ({(len(test_df)/len(df))*100:.1f}%)")

    # Create Torch datasets
    train_ds = PitchDataset(train_df, video_dir, yolo_model_dir, mode='train')
    val_ds = PitchDataset(val_df, video_dir, yolo_model_dir, mode='train') # We will still calculate loss of these so we send them in training mode
    test_ds = PitchDataset(test_df, video_dir, yolo_model_dir, mode='train') # We will still calculate loss of these so we send them in training mode

    # Create Data Loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# Updated Validate function to handle 3 outputs
def validate(model, loader, device):
    model.eval()
    total_loss = 0
    criterion_reg = nn.HuberLoss(delta=1.0)
    criterion_cls = nn.BCEWithLogitsLoss()
    # FIX: Added label smoothing for validation consistency
    criterion_zone = nn.CrossEntropyLoss(label_smoothing=0.1) 
    
    w_reg = 1.0
    w_cls = 0.2
    w_zone = 0.2
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating", leave=False)
        for batch in pbar:
            traj = batch['trajectory'].to(device)
            phys = batch['physics'].to(device)
            target_coords = batch['labels'].to(device)
            target_class = batch['class_label'].to(device)
            target_zone = batch['zone_label'].to(device)
            
            pred_coords, pred_logits, pred_zone_logits = model(traj, phys)
            
            loss_coords = criterion_reg(pred_coords, target_coords)
            loss_class = criterion_cls(pred_logits.squeeze(), target_class)
            loss_zone = criterion_zone(pred_zone_logits, target_zone)
            
            loss = (w_reg * loss_coords) + (w_cls * loss_class) + (w_zone * loss_zone)
            total_loss += loss.item()
            
    return total_loss / len(loader)

def main():
    print("Starting Training Script...")
    root_path = Path.cwd()
    root_path = root_path / "competition_folder/baseball-pitch-tracking/kaggle_dataset"

    train_video_dir = root_path / "train_trimmed"
    data_path = root_path / "data"
    train_csv_path = data_path / "train_ground_truth.csv"

    final_test_video_dir = root_path / "test"
    final_test_csv_path = data_path / "test_features.csv"

    print(train_video_dir)
    print(train_csv_path)

    

    yolo_model_path = "./competition_folder/model_output/ball_finetune/weights/best.pt"

    train_dl, val_dl, test_dl = create_data_loaders(train_csv_path, train_video_dir, yolo_model_path, batch_size=4)


    YOLO_PATH =  "./competition_folder/model_output/ball_finetune/weights/best.pt"
    BATCH_SIZE = 4
    ACCUMULATION_STEPS = 8  # Effective Batch Size = 4 * 8 = 32
    LEARNING_RATE = 1e-4    # Lower, constant LR is safer
    NUM_EPOCHS = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Enable Anomaly Detection (Slows down training but finds the exact error source)
    torch.autograd.set_detect_anomaly(True)

    model = MultiTaskPitchModel(physics_dim=9).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Define 3 Losses
    criterion_reg = nn.HuberLoss(delta=1.0)
    criterion_cls = nn.BCEWithLogitsLoss()
    
    # FIX: LABEL SMOOTHING (The "Garbage Prevention" trick)
    # This prevents the model from becoming overconfident on ambiguous zones,
    # which keeps the gradients cleaner for the other tasks.
    criterion_zone = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Loss Weights
    w_reg = 1.0
    w_cls = 0.2
    w_zone = 0.2

    history = {'train_loss_step': [], 'val_loss': [], 'epoch': []}
    print("\nStarting Training with Decoupled Heads & Label Smoothing...")

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        halfway_point = len(train_dl) // 2
        
        pbar = tqdm(enumerate(train_dl), total=len(train_dl), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        optimizer.zero_grad()
        
        for i, batch in pbar:
            traj = batch['trajectory'].to(device)
            phys = batch['physics'].to(device)
            target_coords = batch['labels'].to(device)
            target_class = batch['class_label'].to(device)
            target_zone = batch['zone_label'].to(device)
            
            # Forward 3 outputs
            pred_coords, pred_logits, pred_zone_logits = model(traj, phys)
            
            # Calculate 3 losses
            loss_coords = criterion_reg(pred_coords, target_coords)
            loss_class = criterion_cls(pred_logits.squeeze(), target_class)
            loss_zone = criterion_zone(pred_zone_logits, target_zone)
            
            # Weighted Sum
            loss = (w_reg * loss_coords) + (w_cls * loss_class) + (w_zone * loss_zone)
            
            # Accumulation
            loss = loss / ACCUMULATION_STEPS
            loss.backward()
            
            if (i + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            current_loss = loss.item() * ACCUMULATION_STEPS
            running_loss += current_loss
            
            pbar.set_postfix({'loss': current_loss})
            history['train_loss_step'].append(current_loss)

            if i == halfway_point:
                val_loss = validate(model, val_dl, device)
                history['val_loss'].append(val_loss)
                history['epoch'].append(epoch + 0.5)
                print(f"\n[Epoch {epoch+1} Mid] Val Loss: {val_loss:.4f}")
                model.train()

        val_loss = validate(model, val_dl, device)
        history['val_loss'].append(val_loss)
        history['epoch'].append(epoch + 1.0)
        
        avg_train_loss = running_loss / len(train_dl)
        print(f"\nEpoch {epoch+1} Results: Train Loss: {avg_train_loss:.4f} | End Val Loss: {val_loss:.4f}")

    print("\nTraining Complete.")

    save_path = f"competition_folder/multitask_model_with_zones_{NUM_EPOCHS}epochs.pth"
    print(f"Saving model to {save_path}")
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    main()