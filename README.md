# alpha-zero-cpp
Alpha Zero implementation that combines the speed of C++ and flexbility of Python. 


Games, Selfplay and Monte Carlo Tree Search (MCTS) is done in C++. 
Neural network architecture implementation and training is done in Python using Pytorch.

This is implementation is similar to [Polygames](https://github.com/facebookincubator/Polygames). 

## Table of Contents

- [alpha-zero-cpp](#alpha-zero-cpp)
  - [Table of Contents](#table-of-contents)
  - [Quick Start](#quick-start)
    - [Requirements](#requirements)
	- [Running](#running)
    - [Directory Structure](#directory-structure)


## Quick Start

### Requirements
Atleast Python >3.7.0 and C++14 is needed.

To install python requirements, run:

```[bash]
pip install -r requirements.txt
```

C++ libraries are defined in the file cc/Dependencies.cmake which is used when you build using cmake.

To build using cmake, run the following:
```[bash]
mkdir build && cd build && cmake --build .
```

The first time running build will probabily be slow as several libraries are being downloaded and built. If you get an error, when installing libtorch the first time you build, run the command again.

### Running
To run you can simply, run the following script:  
```[bash]
python run_az.py
```

This script launches three processes: one for selfplay, one for training and one for testing.
You can find all relevant parameters defined in that script and change you as you wish. 
There are other paramaters defined inside the respective executables, but the default should be enough.

You can also optimize for certain parameters by running the script:
```[bash]
python optimize.py
```

However, you need a dataset for that, you can do that either by running the generated.cc executable or selfplay.cc executable.

### Directory Structure

The code is organized as follows:

```[txt]
.
├── LICENSE
├── README.md
├── __init__.py
├── cc
│	├── games
│	│	├── ConnectFour.cc/h 	# Connect Four implementation
│	│	├── Game.cc/h 			# Game class. All implemented games must inherent this class
│	│	└── Tictactoe.cc/h  	# Tictactoe implementation
│	├── mcts
│	│	├── GameState.cc/h      # Class that represents a game state (board position or a node) used in MCTS 
│	│	└──	MCTS.cc/h 			# MCTS implementation. Several options, such as: parallelized, use a NN for node evaluation etc..
│	├── nn
│	│	├── NNObserver.cc/h 	# Observer to detect when the neural net weights need to be updated (No longer used)
│	│	├──	NNUtils.h 			# Utils to convert the neural net inputs and outputs to correct format
│	│	├──	NNWrapper.cc/h    	# Class that contains the NN, this is used in MCTS
│	│	└──	utils.h 			# Utils to convert tensors between libraries 
│	├── player
│	│	└── Player.cc/h 		# Contains implementation of several types of players 
│   ├── game_info.cc 			# Executable to obtain information about a specific game. This is used to be able to build the network 
│   ├── generate.cc 			# Executable that generates games through selfplay using players that play randomly
│   ├── play_agaisnt.cc 		# Executable that pits two players agaisnt each other
│   └── selfplay.cc             # Executable that generates games through selfplay the Alpha Zero way
├── py
│   ├── utils/utils.py          # Plots statistics from the games generated    
│   ├── GameSampler.py          # Several samplers that sample games for training    
│   └── NN.py              		# Neural Network implementation and wrapper for training
├── requirements.txt
├── optimize.py         		# Optimizes neural networks hyperparameters using bayesian optimization
├── plot.py         			# Plots elo using bayes elo from results obtained from play_agaisnt
├── run_az.py         			# Main script: it launches game generation (selfplay.cc), training (train.py) and testing (play_agaisnt.cc)
├── train.py         			# Training script
└── utils.py         			# Utils used in plot.py
```
