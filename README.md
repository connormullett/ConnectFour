
# Connect Four AI Attempt

Data not inluded for size of repo.
data can be found [here](https://archive.ics.uci.edu/ml/datasets/Connect-4)

# Still a work in progress

# Requires
- pytorch
- pandas
- numpy

# Usage
 - Generate `board.npy` file using `./process_data.py`
 - `board.npy` is used by `generate_data.py` for creating
  the torch Dataset object in the same file

# TODO
 - Game server: requests turn
  takes turn from model,
  saves turn,
  when game is won, give to model

