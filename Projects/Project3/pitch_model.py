import torch
import torch.nn as nn

class MultiTaskPitchModel(nn.Module):
    def __init__(self, physics_dim=9, hidden_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()

        # --- Branch 1: Video Trajectory (Transformer) ---
        # Input: [Batch, 16, 2] (x, y normalized coordinates)
        self.traj_embedding = nn.Linear(2, hidden_dim)

        # Learnable positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 16, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Branch 2: Physics (MLP) ---
        # Input: [Batch, physics_dim]
        # CRITICAL FIX: Changed BatchNorm1d to LayerNorm
        # BatchNorm is unstable with batch_size=4. LayerNorm works perfectly.
        self.physics_mlp = nn.Sequential(
            nn.Linear(physics_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # --- Fusion Layer ---
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.activation = nn.ReLU()

        # --- HEAD 1: Coordinate Regression (Main Task) ---
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        # --- HEAD 2: Strike Classification (Auxiliary Task) ---
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # --- STABILITY FIX: WEIGHT INITIALIZATION ---
        # Apply Xavier/Glorot initialization to all linear layers
        self.apply(self._init_weights)

        # Initialize final projection layers with small weights
        # This ensures predictions start near 0, preventing massive initial MSE loss
        with torch.no_grad():
            self.coord_head[-1].weight.mul_(0.01)
            self.coord_head[-1].bias.zero_()
            self.class_head[-1].weight.mul_(0.01)
            self.class_head[-1].bias.zero_()

    def _init_weights(self, module):
        """
        Xavier Uniform Initialization:
        Keeps the scale of gradients roughly the same in all layers.
        Crucial for preventing exploding gradients in Transformers/Deep Networks.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, traj, physics):
        # 1. Trajectory Branch
        t_emb = self.traj_embedding(traj) + self.pos_encoder
        t_out = self.transformer(t_emb)
        t_feat = t_out[:, -1, :]

        # 2. Physics Branch
        p_feat = self.physics_mlp(physics)

        # 3. Fusion
        combined = torch.cat([t_feat, p_feat], dim=1)
        fused = self.activation(self.fusion(combined))

        # 4. Outputs
        predicted_coords = self.coord_head(fused)
        predicted_logits = self.class_head(fused)

        return predicted_coords, predicted_logits