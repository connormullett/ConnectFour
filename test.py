#!/usr/bin/env python

import torch
import sys
import random
import numpy as np
from train import Net
from connect_four import Game, RED, YELLOW
from train import predict


def request_move(model, board, chance):
  use_random = random.random() < chance
  if use_random:
    prediction = torch.rand(7) 
    return prediction
  else:
    return predict(model, board)


def main():
  net = Net()
  net.load_state_dict(torch.load(PATH))

  g = Game()
  red = True
  moves = 0

  while not g.won():
    board = g.to_tensor(red)

    prediction = request_move(net, board, chance=0.2 if red else 0.0)
    print(prediction)
    max_value = torch.max(prediction)
    move = np.zeros((1, 7))

    for i, element in enumerate(prediction):
      if element == max_value:
        move[:, i] = 1
        prediction[i] = 0.

    column = move.tolist()[0].index(1.)

    for iter in range(7):
      try:
        g.insert(column, RED if red else YELLOW)
        moves += 1
        break
      except Exception as e:
        column = (column + 1) % 7
    else:
      return False, moves
    
    if g.won():
      g.print_board()
      return red, moves

    # switch turn
    if red:
      red = False
    else:
      red = True


if __name__ == '__main__':

  if len(sys.argv) > 2:
    PATH = sys.argv[2]
    net = Net()
    net.load_state_dict(torch.load(PATH))
  else:
    net = Net()

  games = int(sys.argv[1])

  red_wins = 0
  total_moves = 0
  
  for i in range(games+1):
    red_winner, moves = main()
    total_moves += moves 
    if red_winner:
      red_wins += 1
  print(f'{(red_wins/games) * 100}')
  print(f'{total_moves/games}')

