import torch
import torch.nn as nn
import torch.optim as optim

class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()

        # Input Transformation Network
        self.input_transform = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        # Feature Transformation Network
        self.feature_transform = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # Fully Connected Layers for Classification
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 63),  # Output 63 coordinates (21 joints * 3 coordinates)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Using Xavier Uniform initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input Transformation
        x = self.input_transform(x)
        x = torch.max(x, 2, keepdim=True)[0]

        # Feature Transformation
        x = self.feature_transform(x)
        x = torch.max(x, 2, keepdim=True)[0]

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Fully Connected Layers
        x = self.fc(x)

        # Reshape to [batch_size, 3, 21]
        x = x.view(x.size(0), 3, 21)

        return x

# Example usage:
# model = PointNet()
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
