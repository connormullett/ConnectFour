#!/usr/bin/env python3

from train import Net, update_model, predict
import numpy as np
import torch
import random
import connect_four

from datetime import datetime
from connect_four import RED, YELLOW


def request_move(model, board):
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

  # 20% of bar's moves are random
  bars_random_move_probability = 0.2

  foos_turn = True
  game = connect_four.Game()

  while not game.won():

    # encode boards current state to tensor
    board = game.to_tensor()

    while(1):
      # look to create a random move only on bars turn
      if bars_random_move_probability > random.random()\
          and not foos_turn:
        move = np.zeros((1, 7))
        rand = random.randint(0, 7)
        move[0][rand] = 1
        prediction = torch.from_numpy(move).flatten()
      else:
        # bar plays normally
        prediction = request_move(net, board)
        max_value = torch.max(prediction)
        move = np.zeros((1, 7))

      for i, tensor in enumerate(prediction):
        if tensor == max_value:
          move[:, i] = 1
          prediction[i] = 0.

      move = move.tolist()
      column = move[0].index(1.)

      try:
        game.insert(column, RED if foos_turn else YELLOW)
        break
      except Exception as e:
        # zero prediction@column and get new max
        prediction[column] = 0.

      game.print_board()

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
  games = 100

  net = Net()
  for game_num in range(games+1):

    # play the game
    play(net)

  save_model(net)

