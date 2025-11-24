import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeechCommandModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)

        # Автоматичний розрахунок розміру для fc1
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 64, 64)
            out = self.conv2(F.relu(self.conv1(dummy)))
            flat_dim = out.numel()

        self.fc1 = nn.Linear(flat_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

