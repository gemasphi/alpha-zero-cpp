import yaml
import optuna
#from src.AlphaZeroTrainer import AlphaZeroTrainer as az 
from py.NN import NetWrapper
from py.games.Tictactoe import Tictactoe
#from py.Player import * 
#from py.MCTS_virtual_loss import MCTS
import json 
import os
import numpy as np
import torch.optim as optim

#{'lr': 0.012486799182525229, 'wd': 0.015010724465999522}.

GAME_DIR = "cc/build/games/"
SAVE_CHECKPOINT = 100
LOSS_LOG = 10
BATCH_SIZE = 36
GAME_WINDOW = 50

def make_input(history, i):
	player = -1 if i % 2 == 0 else 1 
	return player*np.reshape(history[i], (3,3))       

def make_target(winner, probs, i):
	player = -1 if i % 2 == 0 else 1  
	return probs[i], winner*player 

def sample_batch(batch_size):
	all_games = sorted(
		os.listdir(GAME_DIR), 
		key = lambda f: os.path.getctime("{}/{}".format(GAME_DIR, f))
		)[:GAME_WINDOW]

	game_files = np.random.choice(all_games, size = batch_size)
	games = [json.load(open(GAME_DIR + game_f)) for game_f in game_files]
	game_pos = [(g, np.random.randint(len(g['history']))) for g in games]
	pos = np.array([
		[make_input(g['history'], i), *make_target(g['winner'], g['probabilities'], i)] 
		for (g, i) in game_pos
		])

	return list(pos[:,0]), list(pos[:,1]), list(pos[:,2])

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

game = Tictactoe(**config['GAME'])
#trace_model(game)

nn = NetWrapper(game, **config['NN'])


i = 0
loss = 0
while True:
	#print(batch)
	#print(batch)
	batch = sample_batch(BATCH_SIZE)
	loss += nn.train(batch)
	if i % SAVE_CHECKPOINT == 0 and i != 0:
		print("New model saved")
		nn.save_traced_model(folder = "cc", model_name = "traced_model.pt")

	if i % LOSS_LOG == 0 and i != 0:
		print(loss/LOSS_LOG)
		loss = 0

	i += 1


"""

mcts = MCTS(**config['MCTS'])
#nn.load_model()

alphat = az(nn, game, mcts, **config['AZ'])
loss = alphat.train(lr = 0.01, wd = 0.015)
"""