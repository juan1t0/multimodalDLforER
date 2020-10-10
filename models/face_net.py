import torch
import torch.nn as nn

class Model (nn.Module):
  def __init__(self, inchanels=144, outchanels=26) :
    super(Model, self).__init__()
    self.inchanels = inchanels
    self.outchanels = outchanels

    self.conv_layers = nn.Sequential(
        nn.Conv1d(inchanels, 256, kernel_size=1),
        nn.BatchNorm1d(256),
        nn.ReLU(),

        nn.Conv1d(256, 512, kernel_size=1),
        nn.BatchNorm1d(512),
        nn.ReLU(),

        nn.Conv1d(512, 1024, kernel_size=1),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
 
        nn.MaxPool1d(kernel_size=1, stride=1)
    )
    
    self.lner_layers = nn.Sequential(
        nn.Linear(1024, 1024),
        nn.Linear(1024, 512),
        nn.Linear(512, 256),
        nn.Linear(256, outchanels)
    )
  def forward(self, x):
    x = self.conv_layers(x)
    x = x.view(x.size(0), -1)
    x = self.lner_layers(x)
    return x