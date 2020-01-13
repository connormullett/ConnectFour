
from train import Net, train
import connect_four


def request_move():
  # takes in a board
  # feed to model
  # return a move
  board = request.json

def update_model():
  # takes in a list of moves for winning model
  # updates model with those moves
  move_list = request.json

if __name__ == '__main__':
  app.run(debug=True)

