#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch import optim

from generate_data import ConnectDataset


# 6 * 7 * 3 = 126
# + 1 for player move ??
# = 127
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.a1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
    self.a2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
    self.a3 = nn.Conv2d(16, 32, kernel_size=3, stride=1)

    self.b1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
    self.b2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
    self.b3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)

    self.c1 = nn.Conv2d(64, 128, kernel_size=2, padding=1)
    self.c2 = nn.Conv2d(128, 128, kernel_size=2, padding=1)
    self.c3 = nn.Conv2d(128, 128, kernel_size=2, stride=2)

    self.last = nn.Linear(128, 1)

  def forward(self, x):
    x = x.view(-1, 3, 6, 7)

    x = F.relu(self.a1(x))
    x = F.relu(self.a2(x))
    x = F.relu(self.a3(x))

    x = F.relu(self.b1(x))
    x = F.relu(self.b2(x))
    x = F.relu(self.b3(x))

    x = F.relu(self.c1(x))
    x = F.relu(self.c2(x))
    x = F.relu(self.c3(x))

    # reshape to 8, 128
    x = x.view(-1, 128)
    x = self.last(x)
    x = torch.tanh(x)
    return x


def train(model, dataloader, optimizer, criterion, epochs=5):
  for i, board in enumerate(dataloader):
    print(board[0])
    board = board.to(device)
    out = model(board)
    print(out)
    print(out.shape)
    break


if __name__ == '__main__':
  device = "cuda" if torch.cuda.is_available else "cpu"

  data = ConnectDataset()
  dataloader = DataLoader(data, batch_size=4,
      shuffle=True, num_workers=4)

  # hyper parameters
  learning_rate = 0.01

  net = Net().to(device)
  optimizer = optim.Adam(net.parameters(), lr=learning_rate)
  criterion = torch.nn.CrossEntropyLoss

  train(net, dataloader, optimizer, criterion)

