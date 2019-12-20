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
#import subprocess

GAME_DIR = "games/"
SAVE_CHECKPOINT = 10000000000
LOSS_LOG = 20
BATCH_SIZE = 64
GAME_WINDOW = 35


def make_input(history, i):
	player = -1 if i % 2 == 0 else 1 
	return player*np.array(history[i])       

def make_target(winner, probs, i):
	player = -1 if i % 2 == 0 else 1  
	return probs[i], winner*player 

def sample_batch(batch_size):
	all_games = sorted(
		os.listdir(GAME_DIR), 
		key = lambda f: os.path.getctime("{}/{}".format(GAME_DIR, f))
		)[:GAME_WINDOW]

	game_files = np.random.choice(all_games, size = batch_size)
	#print(game_files)
	games = [json.load(open(GAME_DIR + game_f)) for game_f in game_files]
	game_pos = [(g, np.random.randint(len(g['history']))) for g in games]
	pos = np.array([
		[make_input(g['history'], i), *make_target(g['winner'], g['probabilities'], i)] 
		for (g, i) in game_pos
		])

	return list(pos[:,0]), list(pos[:,1]), list(pos[:,2])
 
def train_az(nn, folder, n_iter = -1, n_epochs = -1):
	nn.load_traced_model("models/traced_model_new.pt")
	i = 0
	loss = 0
	while True:
		batch = sample_batch(BATCH_SIZE)
		loss += nn.train(batch)
		if i % SAVE_CHECKPOINT == 0 and i != 0:
			print("New model saved")
			nn.save_traced_model(folder = folder, model_name = "traced_model_new.pt")

		if i % LOSS_LOG == 0 and i != 0:
			print(loss/LOSS_LOG)
			loss = 0

		i += 1

		if n_epochs > 0 and i > n_epochs:
			nn.save_traced_model(folder = "models", model_name = "traced_model_new.pt")
			#nn.save_model(folder = "models", model_name = "model_new.pt")
			nn.save_traced_model(folder = "models", model_name = "{}_traced_model_new.pt".format(n_iter))

			break

