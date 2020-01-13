#!/usr/bin/env python

import pandas as pd
import torch
import numpy as np


def generate_dataset():
  # take df from csv and output
  # to numpy 'npz' file format
  # to be used by 'generate_data.py'
  df = pd.read_csv('./data/connect-4.data')


  # statuses from player 'x' perspective
  # win = 1, loss = 0, draw = 0
  boards = df.drop(labels=['win'], axis=1)
  classes = df.loc[:, 'win']

  boards_matrix = boards.values

  # the big boi
  # 67557 - 1 boards, 3 channels, 6 rows, 7 columns
  dataset = np.zeros((67556, 3, 6, 7))

  print('parsing games ...')
  for j, game in enumerate(boards_matrix):
    print('game %d of %d' % (j, len(dataset)), end='\r')
    # create 1d matrixes for parsing games
    # to 3 different one-hot channels
    b = np.zeros((1, 42), dtype=np.uint8)
    o = np.zeros((1, 42), dtype=np.uint8)
    x = np.zeros((1, 42), dtype=np.uint8)

    for i, cell in enumerate(game):
      if cell == 'o':
        o[0, i] = 1
      elif cell == 'x':
        x[0, i] = 1
      else:
        b[0, i] = 1

    # reshape the channels
    b = b.reshape((6, 7))
    o = o.reshape((6, 7))
    x = x.reshape((6, 7))

    # add channels to a new board
    new_board = np.stack((b, o, x), axis=0)

    # add new board to the big boi
    dataset[j] = new_board

  np.save('./processed/boards.npy', dataset)

if __name__ == '__main__':
  generate_dataset()

