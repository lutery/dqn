import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np


class GameNetwork(nn.Module):

    def __init__(self, OBS_SHAPE, ACTION_SHAPE, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.feature = nn.Linear(OBS_SHAPE, 1024)
        self.feature_rule = nn.ReLU()
        self.action = nn.Linear(1024, ACTION_SHAPE)

    
    def forward(self, x):
        feature_v = self.feature(x)
        feature_v = self.feature_rule(feature_v)
        pred_action = self.action(feature_v)
        return pred_action

    
if __name__ == "__main__":
    net = GameNetwork(OBS_SHAPE=14, ACTION_SHAPE=100).to("cuda")
    x = torch.rand(128, 14).to("cuda")
    action = net(x)
    print(action.size(), action)