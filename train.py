#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    # 3 channels -> 1 channel
    self.a1 = nn.Conv2d(3, 3, kernel_size=4)
    self.a2 = nn.Conv2d(3, 2, kernel_size=1)
    self.a3 = nn.Conv2d(2, 1, kernel_size=1)

    self.fc = nn.Linear(12, 7)
    self.softmax = nn.Softmax(dim=0)

  def forward(self, x):
    x = x.view(-1, 3, 6, 7)

    x = F.relu(self.a1(x))
    x = F.relu(self.a2(x))
    x = F.relu(self.a3(x))

    x = x.flatten()
    x = self.softmax(self.fc(x))
    return x


# call this until win
# backprop and optimizing comes later
def predict(model, board):
  out = model(board)
  return out


# when a model wins, send the winning moves/preds
# to train the model and repeat the game
def update_model(model, boards, moves, predictions):
  optimizer = optim.Adam(model.parameters(), lr=0.01)
  for i, tensor in enumerate(moves):
    model.zero_grad()

    tensor = torch.tensor(tensor).flatten()
    input = boards[i]

    out = model(input)

    tensor = torch.tensor([x * (i / len(moves)) for x in tensor])
    out.backward(tensor)

    optimizer.step()


if __name__ == '__main__':
  device = "cuda" if torch.cuda.is_available else "cpu"

  learning_rate = 0.01

  net = Net().to(device)
  criterion = torch.nn.MSELoss()

  train(optimizer, criterion)
  torch.save(net.state_dict(), './model.pth')

