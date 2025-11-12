# CS539 Final Project

Kamineni Balaji
Alexander Beck
Brian Berkeley
Nate Schneider

Dataset link: [https://www.kaggle.com/datasets/tonygordonjr/spotify-dataset-2023?select=spotify_data_12_20_2023.csv](https://www.kaggle.com/datasets/tonygordonjr/spotify-dataset-2023?select=spotify_data_12_20_2023.csv)

## Installation

1. `python -m venv .venv`
1. Activate the virtual environment
1. `pip install -r requirements.txt`

TODO:

- [ ] Add installation steps for getting dataset. @Balaji-Kamineni
- [ ] Gradient Decent for max split (tree depth)
- [ ] Gradient Decent for track_popularity threshold
- [ ] Training loop @AlexanderBeck0
- [ ] Create a representative representation of the data for faster training and testing @NateSchneid
- [x] Co-gradient decent - one step in track_popularity, one step in max depth, greedily take which affects accuracy more.
- [ ] Grid search instead of co-gradient decent @DataScienceGorilla
  - Start at 60 if using co-gradient decent, with a hard lower bound of 50 (per definition of "popular")
  - [50, 70] for track_popularity
  - [5, 15] for max depth bounds
- [ ] BY END OF WEEK, have proof of concept of co-gradient decent

Use figures (from gradient decent), train full model

## Usage

TODO:

- [ ] Explain model usage
