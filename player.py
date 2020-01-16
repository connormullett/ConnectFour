#!/usr/bin/env python3

from train import Net, update_model, predict
import sys
import numpy as np
import torch
import random
import connect_four

from datetime import datetime
from connect_four import RED, YELLOW


def request_move(model, board, chance=2):
  use_random = random.random() < chance
  if use_random:
    prediction = torch.rand(7)
    return prediction
  else:
    return predict(model, board)


def save_model(model):
  PATH = './nets/model.pth'
  torch.save(model.state_dict(), PATH)


def view_parameters(layer):
  print(list(layer.parameters()))


def play(net):

  # player foo and bar locked in eternal battle
  foo_moves = []
  bar_moves = []

  foo_preds = []
  bar_preds = []

  foo_boards = []
  bar_boards = []

  foos_turn = True
  game = connect_four.Game()

  while not game.won():

    board = game.to_tensor(foos_turn)

    if foos_turn:
      game.print_board()
      while(1):
        try:
          x = int(input())
          break
        except Exception:
          continue
      one_hot = np.zeros(7)
      one_hot[x] = 1
      prediction = torch.tensor(one_hot)
    else:
      prediction = request_move(net, board)

    max_value = torch.max(prediction)
    move = np.zeros((1, 7))

    for i, element in enumerate(prediction):
      if element == max_value:
        move[:, i] = 1
        prediction[i] = 0.

    column = move.tolist()[0].index(1.)

    for iter in range(7):
      try:
        game.insert(column, RED if foos_turn else YELLOW)
        break
      except Exception as e:
        column = (column + 1) % 7
    else:
      # game.print_board()
      return


    # switch turn
    if foos_turn:
      foo_moves.append(move)
      foo_preds.append(prediction)
      foo_boards.append(board)
      foos_turn = False
    else:
      bar_moves.append(move)
      bar_preds.append(prediction)
      bar_boards.append(board)
      foos_turn = True

  game.print_board()
  # if its no longer foos turn, he won
  good_moves = foo_moves if not foos_turn else bar_moves
  good_preds = foo_preds if not foos_turn else bar_preds
  good_boards = foo_boards if not foos_turn else bar_boards

  bad_moves = foo_moves if foos_turn else bar_moves
  bad_moves = [(move - 1) * -1 for move in bad_moves]

  bad_preds = foo_preds if foos_turn else bar_preds
  bad_boards = foo_boards if foos_turn else bar_boards

  update_model(net, good_boards, good_moves, good_preds)
  update_model(net, bad_boards, bad_moves, bad_preds)


if __name__ == '__main__':
  games = int(sys.argv[1])

  if len(sys.argv) > 2:
    net = Net()
    net.load_state_dict(torch.load(sys.argv[2]))
  else:
    net = Net()

  for game_num in range(1,games+1):
    play(net)

    save_model(net)

