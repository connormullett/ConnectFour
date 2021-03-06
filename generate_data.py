#!/usr/bin/env python

import pandas as pd
import torch
import numpy as np
import process_data

from torch.utils.data import DataLoader, Dataset

device = "cuda" if torch.cuda.is_available else "cpu"

# 6 * 7 + 1
# 42 different positions + 1 state for player turn
class ConnectDataset(Dataset):
  # class for use with CLEAN, PROCESSED data

  def __init__(self):
    # will need to generate classes
    # if going back to values net instead
    # of classifying best column for move
    self.boards = torch.from_numpy(
      np.load('./processed/boards.npy'),
    ).to(torch.float32)
    self.targets = process_data.generate_classes()

  def __len__(self):
    return len(self.boards)
  
  def __getitem__(self, idx):
    x = self.boards[idx]
    y = self.targets[idx]
    return x, y

if __name__ == '__main__':
  data = ConnectDataset()
  print(data[0])

