import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import pandas as pd
import os
import numpy as np
from pathlib import Path

# Import your custom modules
from lstm_dataset import PitchDataset
from lstm_model import HybridPitchModel

# --- Configuration ---
BATCH_SIZE = 4
GRAD_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20

root_path = Path.cwd()
competition_path = root_path / "competition_folder/baseball-pitch-tracking/kaggle_dataset"
data_path = competition_path / "data"

train_video_dir = competition_path / "train_trimmed"
train_csv_path = data_path / "train_ground_truth.csv"

VIDEO_ROOT = str(train_video_dir)
CSV_PATH = str(train_csv_path)
YOLO_PATH = "./competition_folder/model_output/ball_finetune/weights/best.pt"

def validate(model, val_loader, criterion, device, dataset_obj):
    """
    Runs validation loop and calculates Mean Squared Error (Loss)
    and Mean Absolute Error in FEET (interpretable metric).
    """
    model.eval()
    total_loss = 0.0
    total_mae_feet = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            trajectory = batch['trajectory'].to(device)
            physics = batch['physics'].to(device)
            targets = batch['labels'].to(device) # Normalized (plate_x, plate_z)
            
            # Forward Pass
            outputs = model(trajectory, physics)
            
            # Calculate Loss (MSE on normalized values)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # Calculate Interpretable Metric (MAE in Feet)
            # Denormalize predictions and targets
            preds_feet = dataset_obj.denormalize_targets(outputs)
            targets_feet = dataset_obj.denormalize_targets(targets)
            
            # MAE calculation (L1 distance)
            mae = np.mean(np.abs(preds_feet - targets_feet))
            total_mae_feet += mae
            
            num_batches += 1
            
    avg_loss = total_loss / num_batches
    avg_mae = total_mae_feet / num_batches
    return avg_loss, avg_mae

def main():
    # 1. Device Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # 2. Prepare Data
    # Assuming CSV exists. If running blindly, we mock a DataFrame or load real one.
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        
        # Split DF (80/20)
        train_size = int(0.8 * len(df))
        val_size = len(df) - train_size
        train_df = df.iloc[:train_size].reset_index(drop=True)
        val_df = df.iloc[train_size:].reset_index(drop=True)
        
        # Create Train Dataset (Calculates Stats)
        train_dataset = PitchDataset(train_df, VIDEO_ROOT, YOLO_PATH, mode='train')
        
        # Save stats to share with val set
        stats = train_dataset.get_stats()
        
        # Create Val Dataset (Uses Train Stats)
        val_dataset = PitchDataset(val_df, VIDEO_ROOT, YOLO_PATH, mode='train', stats=stats)
        
        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
    else:
        print(f"Error: {CSV_PATH} not found. Please provide actual data path.")
        return

    # 3. Model Setup
    model = HybridPitchModel(physics_dim=13).to(device)
    
    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss() # Standard regression loss

    # 4. Training Loop
    print("Starting training...")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad() # Reset gradients at start of epoch
        
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
            
            # Normalize loss by accumulation steps
            # (Because gradients sum up, we average them over the steps)
            loss = loss / GRAD_ACCUMULATION_STEPS
            
            # Backward Pass
            loss.backward()
            
            # Gradient Accumulation Step
            if (batch_idx + 1) % GRAD_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Update Progress Bar (multiply loss back for display)
            epoch_loss += loss.item() * GRAD_ACCUMULATION_STEPS
            progress_bar.set_postfix(loss=loss.item() * GRAD_ACCUMULATION_STEPS)
            
        # Handle leftover gradients if dataset size isn't divisible by accum steps
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
        
        # Save Checkpoint
        torch.save(model.state_dict(), f"hybrid_pitch_model_ep{epoch+1}.pth")

    print("Training Complete.")

if __name__ == "__main__":
    main()