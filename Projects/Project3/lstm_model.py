import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridPitchModel(nn.Module):
    def __init__(self, physics_dim=13, seq_len=16, hidden_dim=64):
        """
        Args:
            physics_dim (int): Number of static physics features (13 in dataset.py).
            seq_len (int): Length of the video sequence (16 frames).
            hidden_dim (int): Internal dimension for LSTM and Dense layers.
        """
        super(HybridPitchModel, self).__init__()
        
        # --- Branch 1: Visual Trajectory (LSTM) ---
        # Input: (Batch, Seq_Len, 2) -> (x, y) relative coordinates
        self.lstm = nn.LSTM(input_size=2, 
                            hidden_size=hidden_dim, 
                            num_layers=1, 
                            batch_first=True)
        
        # --- Branch 2: Physics Context (MLP) ---
        # Input: (Batch, physics_dim)
        self.physics_mlp = nn.Sequential(
            nn.Linear(physics_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2), # Dropout to prevent overfitting on small dataset
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # --- Fusion Head ---
        # Concatenates LSTM output (hidden_dim) + Physics output (hidden_dim)
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            # Output: 2 values (Normalized plate_x, plate_z)
            nn.Linear(32, 2)
        )

    def forward(self, trajectory, physics):
        """
        Args:
            trajectory: Tensor (Batch, 16, 2)
            physics: Tensor (Batch, 13)
        """
        # 1. Process Video Branch
        # lstm_out: (Batch, Seq_Len, Hidden)
        # We only care about the final hidden state or the last output time step
        lstm_out, (hn, cn) = self.lstm(trajectory)
        
        # Take the output of the last time step
        visual_features = lstm_out[:, -1, :] 
        
        # 2. Process Physics Branch
        physics_features = self.physics_mlp(physics)
        
        # 3. Fusion
        combined = torch.cat((visual_features, physics_features), dim=1)
        
        # 4. Regression Prediction
        prediction = self.fusion_layer(combined)
        
        return prediction