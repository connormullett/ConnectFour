#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch import optim

from generate_data import ConnectDataset


class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.a1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
    self.a2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
    self.a3 = nn.Conv2d(3, 3, kernel_size=3, stride=1)

    self.last = nn.Linear(60, 1)
    self.out = nn.Sigmoid()

  def forward(self, x):
    x = x.view(-1, 3, 6, 7)

    x = F.relu(self.a1(x))
    x = F.relu(self.a2(x))
    x = F.relu(self.a3(x))

    # reshape to 8, 128
    x = x.view(-1, 60)
    x = self.last(x)
    x = self.out(x)
    return x


def train(model, dataloader, optimizer, criterion, epochs=5):
  model.train()

  for i, (board, target) in enumerate(dataloader):
    optimizer.zero_grad()
    board = board.to(device)
    print(board)
    out = model(board)
    print(out)
    print(out.shape)

    break
    loss = criterion(out, target)
    loss.backward()
    optimizer.step()


if __name__ == '__main__':
  device = "cuda" if torch.cuda.is_available else "cpu"

  data = ConnectDataset()
  dataloader = DataLoader(data, shuffle=True, num_workers=4)

  # hyper parameters
  learning_rate = 0.01

  net = Net().to(device)
  optimizer = optim.Adam(net.parameters(), lr=learning_rate)
  criterion = torch.nn.MSELoss()

  train(net, dataloader, optimizer, criterion)

