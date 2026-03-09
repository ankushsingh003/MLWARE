import torch
import torch.nn as nn
import torchvision.models as models

class FrameReorderingModel(nn.Module):
    def __init__(self, hidden_dim=256, n_heads=8, num_layers=4, dropout=0.1):
        super(FrameReorderingModel, self).__init__()
        
        # 1. Feature Extractor for individual frames (ResNet18)
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Remove the final fully connected layer and pooling to retain spatial info if needed, 
        # but for simplicity, we keep avgpool and flatten
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = resnet.fc.in_features  # 512 for ResNet18
        
        # 2. Sequence Modeling (Transformer Encoder)
        self.project_in = nn.Linear(self.feature_dim, hidden_dim)
        
        # Transformer Encoder expects inputs of shape (S, N, E) if batch_first=False
        # We'll use batch_first=True -> (N, S, E)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=n_heads, 
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Prediction Head (Scoring)
        # We output a single score per frame. Sorting the frames by their scores gives the predicted order.
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
        
        # Reshape to process all frames independently
        x = x.view(B * S, C, H, W)
        
        # Extract features
        features = self.feature_extractor(x)  # (B*S, 512, 1, 1)
        features = features.view(B, S, self.feature_dim)
        
        # Project dimension
        features = self.project_in(features)
        
        # Sequence modeling
        # Provide relative/absolute positional encoding if necessary, 
        # but since frames are scrambled, we want the model to learn 
        # the temporal relationships from the visual content and predict temporal indices.
        transformed_features = self.transformer(features)
        
        # Predict scores
        scores = self.scorer(transformed_features).squeeze(-1)  # (B, S)
        
        return scores
