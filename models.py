import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, num_classes = 10, input_channel = 3):
        super(CNNModel, self).__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(input_channel, 128, 3),
            nn.Conv2d(128, 128, 3),
            nn.Conv2d(128, 128, 3),
            nn.MaxPool2d(2, stride = 2),
            nn.Dropout(p = 0.25),
            nn.Conv2d(128, 256, 3),
            nn.Conv2d(256, 256, 3),
            nn.Conv2d(256, 256, 3),
            nn.MaxPool2d(2, stride = 2),
            nn.Dropout(p = 0.25),
            nn.Conv2d(256, 512, 3),
            nn.Conv2d(512, 256, 3),
            nn.Conv2d(256, 128, 3),
            nn.AdaptiveAvgPool2d((1,1)),
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(-1, 128)
        out = self.classifier(feature)
        return out