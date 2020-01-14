#! /usr/bin/env python3
from itertools import groupby, chain
import torch
import numpy as np


NONE = '.'
RED = 'R'
YELLOW = 'Y'


def diagonal_pos(matrix, cols, rows):
  for di in ([(j, i - j) for j in range(cols)] for i in range(cols + rows -1)):
    yield [matrix[i][j] for i, j in di if i >= 0 and j >= 0 and i < cols and j < rows]


def diagonal_neg(matrix, cols, rows):
  for di in ([(j, i - cols + j + 1) for j in range(cols)] for i in range(cols + rows - 1)):
    yield [matrix[i][j] for i, j in di if i >= 0 and j >= 0 and i < cols and j < rows]


class Game:
  def __init__ (self, cols=7, rows=6, required_to_win=4):
    self.cols = cols
    self.rows = rows
    self.win = required_to_win
    self.board = [[NONE] * rows for _ in range(cols)]

  def to_tensor(self):
    # self.board = 7,6
    red = np.zeros(42)
    yellow = np.zeros(42)
    blank = np.zeros(42)

    board = np.asarray(self.board).reshape(42)
    for i, cell in enumerate(board[0]):
      if cell == 'R':
        red[i] = 1
      elif cell == 'Y':
        yellow[i] = 1
      else:
        blank[i] = 1

    red = red.reshape((6, 7))
    yellow = yellow.reshape((6, 7))
    blank = blank.reshape((6, 7))

    out = np.stack((blank, yellow, red), axis=0)
    return torch.from_numpy(out).to(torch.float32)

  def insert(self, column, color):
    c = self.board[column]
    if c[0] != NONE:
      raise Exception('Column is full')

    i = -1
    while c[i] != NONE:
      i -= 1
    c[i] = color

    self.won()

  def won(self):
    w = self.get_winner()
    if w:
      return True

  def get_winner(self):
    lines = (
      self.board, # columns
      zip(*self.board), # rows
      diagonal_pos(self.board, self.cols, self.rows), # positive diagonals
      diagonal_neg(self.board, self.cols, self.rows) # negative diagonals
    )

    for line in chain(*lines):
      for color, group in groupby(line):
        if color != NONE and len(list(group)) >= self.win:
          return color

  def print_board(self):
    print('  '.join(map(str, range(self.cols))))
    for y in range(self.rows):
      print('  '.join(str(self.board[x][y]) for x in range(self.cols)))
    print()


if __name__ == '__main__':
  g = Game()
  # turn = RED
  # while True:
  #   g.printBoard()
  #   row = input('{}\'s turn: '.format('Red' if turn == RED else 'Yellow'))
  #   g.insert(int(row), turn)
  #   turn = YELLOW if turn == RED else RED
  board = g.to_tensor()
  g.insert(1, RED)
  g.insert(5, YELLOW)
  print(g.board)
