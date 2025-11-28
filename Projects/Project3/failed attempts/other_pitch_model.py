import torch
import torch.nn as nn

class MultiTaskPitchModel(nn.Module):
    def __init__(self, physics_dim=9, hidden_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        
        # --- Branch 1: Video Trajectory ---
        self.traj_embedding = nn.Linear(2, hidden_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, 16, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4, dropout=dropout, batch_first=True, norm_first=True)
        self.traj_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # --- Branch 2: Physics ---
        self.physics_dim = physics_dim
        self.feat_embeddings = nn.ModuleList([nn.Linear(1, hidden_dim) for _ in range(physics_dim)])
        physics_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4, dropout=dropout, batch_first=True, norm_first=True)
        self.physics_transformer = nn.TransformerEncoder(physics_encoder_layer, num_layers=num_layers)
        
        # --- DECOUPLED FUSION LAYERS (The Fix) ---
        # Instead of one shared fusion layer, we create separate ones.
        # This prevents "Negative Transfer" where the noisy regression task 
        # confuses the classification task (and vice versa).
        
        # 1. Fusion for Regression (Needs precision)
        self.fusion_coord = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. Fusion for Classification (Needs abstraction)
        self.fusion_class = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # --- HEADS ---
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2) 
        )
        
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1) 
        )
        
        self.zone_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 13) 
        )

        # --- Initialization ---
        self.apply(self._init_weights)
        with torch.no_grad():
            self.coord_head[-1].weight.mul_(0.01)
            self.coord_head[-1].bias.zero_()
            self.class_head[-1].weight.mul_(0.01)
            self.class_head[-1].bias.zero_()
            self.zone_head[-1].weight.mul_(0.01)
            self.zone_head[-1].bias.zero_()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, traj, physics):
        t_emb = self.traj_embedding(traj) + self.pos_encoder 
        t_out = self.traj_transformer(t_emb)
        t_feat = t_out[:, -1, :] 
        
        embeddings = []
        for i, layer in enumerate(self.feat_embeddings):
            col = physics[:, i].unsqueeze(1)
            embeddings.append(layer(col).unsqueeze(1))
        p_seq = torch.cat(embeddings, dim=1)
        p_out = self.physics_transformer(p_seq)
        p_feat = p_out.mean(dim=1) 
        
        # Combine features once
        combined = torch.cat([t_feat, p_feat], dim=1)
        
        # --- BRANCHED PATHS ---
        # Each head gets its own interpretation of the combined features
        
        # Path A: Coordinates
        feat_coord = self.fusion_coord(combined)
        predicted_coords = self.coord_head(feat_coord)
        
        # Path B: Classification (Shared for Strike/Ball and Zone)
        # These tasks are highly related, so they can share a fusion layer
        feat_class = self.fusion_class(combined)
        predicted_logits = self.class_head(feat_class)
        predicted_zone_logits = self.zone_head(feat_class)
        
        return predicted_coords, predicted_logits, predicted_zone_logits