import torch
import torch.nn as nn
import torchvision.models as models

class FrameReorderingModel(nn.Module):
    def __init__(self, hidden_dim=256, n_heads=8, num_layers=4, dropout=0.1):
        super(FrameReorderingModel, self).__init__()
        
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = resnet.fc.in_features  # 512 for ResNet18
        
        self.project_in = nn.Linear(self.feature_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=n_heads, 
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        """
        x: (batch_size, num_frames, C, H, W)
        Returns:
        scores: (batch_size, num_frames)
        """
        B, S, C, H, W = x.shape
        
        x = x.view(B * S, C, H, W)
        
        features = self.feature_extractor(x)  # (B*S, 512, 1, 1)
        features = features.view(B, S, self.feature_dim)
        
        features = self.project_in(features)
        
        transformed_features = self.transformer(features)
        
        scores = self.scorer(transformed_features).squeeze(-1)  # (B, S)
        
        return scores
