import torch
import torch.nn as nn
from torchvision.models import resnet18


class VisualEncoder(nn.Module):
    def __init__(self):
        super(VisualEncoder, self).__init__()
        self.cnn = resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.cnn(x)
        features = features.view(batch_size, seq_len, -1)
        return features

class MonocularVIOModel(nn.Module):
    def __init__(self, cnn_feature_dim=512, imu_dim=6, hidden_dim=256):
        super(MonocularVIOModel, self).__init__()
        self.visual_encoder = VisualEncoder()
        self.lstm = nn.LSTM(input_size=cnn_feature_dim + imu_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=2, 
                            batch_first=True,bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, 7)

    def forward(self, images, imu_data):
        visual_features = self.visual_encoder(images)
        fused_features = torch.cat([visual_features, imu_data], dim=-1)
        lstm_out, _ = self.lstm(fused_features)
        poses = self.fc(lstm_out)
        return poses
