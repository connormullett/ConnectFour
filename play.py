#!/usr/bin/env python3

from train import Net, update_model, predict
import sys
import numpy as np
import torch
import random
import connect_four

from datetime import datetime
from connect_four import RED, YELLOW


def request_move(model, board, chance=0.2):
  use_random = random.random() < chance
  if use_random:
    prediction = torch.rand(7) 
    return prediction
  else:
    return predict(model, board)


def save_model(model):
  current = datetime.now().strftime("%H-%M-%S")
  PATH = './nets/model-%s.pth' % current
  torch.save(model.state_dict(), PATH)


def view_parameters(layer):
  print(list(layer.parameters()))


def play(net):

  # player foo and bar locked in eternal battle
  foo_moves = []
  bar_moves = []

  foo_preds = []
  bar_preds = []

  tensor_boards = []
  moves = []

  foos_turn = True
  game = connect_four.Game()

  # move

  while not game.won():

    board = game.to_tensor(foos_turn)

    prediction = request_move(net, board, chance=1 if foos_turn else 0.0)
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
      game.print_board()
      return


    # switch turn
    if foos_turn:
      foo_moves.append(move)
      foo_preds.append(prediction)
      foos_turn = False
    else:
      bar_moves.append(move)
      bar_preds.append(prediction)
      foos_turn = True

    moves.append(move)
    tensor_boards.append(board)

  game.print_board()
  # if its no longer foos turn, he won
  good_moves = foo_moves if not foos_turn else bar_moves
  good_preds = foo_preds if not foos_turn else bar_preds

  # send moves and predictions to update the model
  update_model(net, tensor_boards, moves, good_moves, good_preds)


if __name__ == '__main__':
  games = 1000

  # net = Net()
  games = int(sys.argv[1])
  if len(sys.argv) > 2:
    PATH = sys.argv[2]
    net = Net()
    net.load_state_dict(torch.load(PATH))
  else:
    net = Net()

  for game_num in range(games+1):

    # play the game
    play(net)

  save_model(net)

