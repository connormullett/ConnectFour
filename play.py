#!/usr/bin/env python3

from train import Net, update_model, predict
import numpy as np
import torch
import connect_four

from datetime import datetime
from connect_four import RED, YELLOW


def request_move(model, board):
  return predict(model, board)


def save_model(model):
  current = datetime.now().strftime("%H-%M-%S")
  PATH = './nets/model-%s.pth' % current
  torch.save(model.state_dict(), PATH)


def view_parameters(model):
  for param in model.parameters():
    print(param.data, param.size())


def play(net):

  # player foo and bar locked in eternal battle
  foo_moves = []
  bar_moves = []

  foo_preds = []
  bar_preds = []

  tensor_boards = []
  moves = []

  # foo=0, bar=1
  foos_turn = True
  game = connect_four.Game()

  while not game.won():

    # encode boards current state to tensor
    board = game.to_tensor()

    # get a prediction/move
    prediction = request_move(net, board)

    while(1):
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
        game.print_board()
        break
      except Exception as e:
        # zero prediction@column and get new max
        prediction[column] = 0.

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


  # if its no longer foos turn, he won
  good_moves = foo_moves if not foos_turn else bar_moves
  good_preds = foo_preds if not foos_turn else bar_preds

  # send moves and predictions to update the model
  update_model(net, tensor_boards, moves, good_moves, good_preds)


if __name__ == '__main__':
  games = 10

  for game_num in range(games+1):
    net = Net()
    play(net)

    # save every 5 games
    if game_num % 5 == 0:
      save_model(net)

