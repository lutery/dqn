import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np


class CriticNetwork(nn.Module):

    def __init__(self, OBS_SHAPE, ACTION_SHAPE, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.feature = nn.Linear(OBS_SHAPE, 1024)
        self.feature_rule = nn.ReLU()

        self.value = nn.Linear(1024 + ACTION_SHAPE, 1)
    
    def forward(self, x, a):
        feature_v = self.feature(x)
        feature_v = self.feature_rule(feature_v)
        x = torch.cat((feature_v, a), dim=1)
        pred_value = self.value(x)
        return pred_value

    
if __name__ == "__main__":
    net = CriticNetwork(OBS_SHAPE=14, ACTION_SHAPE=100).to("cuda")
    x = torch.rand(128, 14).to("cuda")
    a = torch.rand(128, 100).to("cuda")
    value = net(x, a)
    print(value.size(), value)