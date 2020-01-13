#!/usr/bin/env python3

from train import Net, update_model, predict
import numpy as np
import torch
import connect_four
from connect_four import RED, YELLOW



def request_move(model, board):
  return predict(model, board)


def update_model():
  # takes in a list of moves for winning model
  # updates model with those moves
  pass


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
    max_value = torch.max(prediction)
    move = np.zeros((1, 7))

    for i, tensor in enumerate(prediction):
      if tensor == max_value:
        move[:, i] = 1

    # insert the piece
    # get column num and insert at column
    move = move.tolist()
    column = move[0].index(1.)

    if foos_turn:
      game.insert(column, RED)
    else:
      game.insert(column, YELLOW)

    print('pred: ', prediction)
    print('move: ', move)
    # game.print_board()
    print(game.to_tensor())
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


if __name__ == '__main__':
  play()

