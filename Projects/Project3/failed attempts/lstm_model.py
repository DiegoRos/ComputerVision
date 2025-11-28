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
                            num_layers=2, # Increased depth to 2
                            dropout=0.2,  # Dropout between LSTM layers
                            batch_first=True)
        
        # Layer Norm is better than Batch Norm for small batch sizes (Batch=4)
        self.ln_visual = nn.LayerNorm(hidden_dim)

        # --- Branch 2: Physics Context (MLP) ---
        # Input: (Batch, physics_dim)
        self.physics_mlp = nn.Sequential(
            nn.Linear(physics_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),   # Stabilizes input distribution
            nn.LeakyReLU(0.1),          # Leaky ReLU prevents dead neurons
            nn.Dropout(0.3),            # Increased dropout
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1)
        )
        
        # --- Fusion Head ---
        # Concatenates LSTM output (hidden_dim) + Physics output (hidden_dim)
        # self.fusion_layer = nn.Sequential(
        #     nn.Linear(hidden_dim * 2, hidden_dim),
        #     nn.LayerNorm(hidden_dim),
        #     nn.LeakyReLU(0.1),
        #     nn.Dropout(0.3),
        #     nn.Linear(hidden_dim, 64),
        #     nn.LeakyReLU(0.1),
        #     # Output: 2 values (Normalized plate_x, plate_z)
        #     nn.Linear(64, 2)
        # )
        
        # ADD specific heads
        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3)
        )

        # Head 1: Regression (Original goal)
        self.reg_head = nn.Linear(hidden_dim, 2) # plate_x, plate_z

        # Head 2: Binary Classification (Strike vs Ball)
        self.cls_head = nn.Linear(hidden_dim, 1) # Logits for BCEWithLogitsLoss

        # Head 3: Zone Classification (Optional, 14 zones)
        self.zone_head = nn.Linear(hidden_dim, 15) # Zones 1-14 (0 reserved)

                # Initialize Weights explicitly to avoid starting with 0 variance
        self._init_weights()

    def _init_weights(self):
        """
        Kaiming Initialization for Relu layers, Xavier for others.
        This breaks the symmetry and prevents the model from predicting '0' everywhere at start.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)

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
        visual_features = self.ln_visual(visual_features)
        
        # 2. Process Physics Branch
        physics_features = self.physics_mlp(physics)
        
        # 3. Fusion
        combined = torch.cat((visual_features, physics_features), dim=1)
    
        # Shared processing
        shared_out = self.shared_layer(combined)
        
        # Independent predictions
        pred_coords = self.reg_head(shared_out)
        pred_class = self.cls_head(shared_out)
        pred_zone = self.zone_head(shared_out)
        
        return pred_coords, pred_class, pred_zone