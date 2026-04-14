import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool3d(2)

        self.fc = nn.Linear(128 * 6 * 6 * 6, 128)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 128 * 6 * 6 * 6)

        self.deconv1 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.deconv2 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.deconv3 = nn.ConvTranspose3d(32, 1, 2, stride=2)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 6, 6, 6)

        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        return torch.sigmoid(self.deconv3(x))


# 🔐 HE-compatible classifier (DO NOT CHANGE)
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 2)

    def forward(self, z):
        return self.fc(z)