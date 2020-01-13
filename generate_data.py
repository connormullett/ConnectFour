#!/usr/bin/env python

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# 6 * 7 + 1
# 42 different positions + 1 state for player turn
class ConnectDataset(Dataset):
  # class for use with CLEAN, PROCESSED data

  def __init__(self):
    # 42 features and 1 class (win/loss)
    # need to convert each board to
    # 3 channels (piece locations)
    # 6 rows and 7 columns

    # this will have to change to prcessed data dir
    self.frame = pd.read_csv('./processed/boards.npy')

  def __len__(self):
    return len(self.frame)
  
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    row = self.frame.iloc[idx, 1:]
    return row


if __name__ == '__main__':
  data = ConnectDataset()
  
  print(data[0])

