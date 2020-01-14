#!/usr/bin/env python3

from train import Net, update_model, predict
import numpy as np
import torch
import connect_four
from connect_four import RED, YELLOW



def request_move(model, board):
  return predict(model, board)


def save_model(model):
  PATH = './nets/model.pth'
  torch.save(model.state_dict(), PATH)


def play():
  net = Net()

  # player foo and bar locked in eternal battle
  foo_moves = []
  bar_moves = []

  foo_preds = []
  bar_preds = []

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
        break
      except Exception as e:
        # zero prediction@column and get new max
        prediction[column] = 0.


    # print('pred: ', prediction)
    # print('move: ', move)
    # print(game.to_tensor())
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


  # if its no longer foos turn, he won
  if not foos_turn:
    # send moves and predictions to update the model
    print('foo won')
    update_model(net, foo_moves, foo_preds)
  else:
    print('bar won')
    update_model(net, bar_moves, bar_preds)


if __name__ == '__main__':
  play()

