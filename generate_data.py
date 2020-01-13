#!/usr/bin/env python

import pandas as pd
import torch
import numpy as np

from torch.utils.data import DataLoader, Dataset


# 6 * 7 + 1
# 42 different positions + 1 state for player turn
class ConnectDataset(Dataset):
  # class for use with CLEAN, PROCESSED data

  def __init__(self):
    self.boards = np.load('./processed/boards.npy')

  def __len__(self):
    return len(self.frame)
  
  def __getitem__(self, idx):
    return self.boards[idx]


if __name__ == '__main__':
  data = ConnectDataset()
  print(data[0])

