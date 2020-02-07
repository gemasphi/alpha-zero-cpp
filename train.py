import yaml
from py.NN import NetWrapper
import json 
import os
import numpy as np
import torch.optim as optim
#import subprocess
import sys
import time
import pandas as pd
import argparse

GAME_DIR = "cc/build/temp/games/"
SAVE_CHECKPOINT = 1000
TEST_SAVE_CHECKPOINT = 5000
LOSS_LOG = 20
BATCH_SIZE = 64
GAME_WINDOW = 0.33

def make_input(history, i, t):
	player = -1 if i % 2 == 0 else 1
	history = np.array(history)
	size = (t, *(history[0].shape))
	history = np.append(np.zeros(size), history, axis = 0)
	return player*history[i: i + t] 

def make_target(winner, probs, i):
	player = -1 if i % 2 == 0 else 1  
	return probs[i], winner*player 

def sample_batch(batch_size):
	n_games = len(os.listdir(GAME_DIR)) 
	all_games = sorted(
		os.listdir(GAME_DIR), 
		key = lambda f: os.path.getctime("{}/{}".format(GAME_DIR, f))
		)[:round(n_games*GAME_WINDOW)]

	game_size = [os.stat(GAME_DIR + file).st_size for file in all_games]
	game_files = np.random.choice(all_games, size = batch_size,  p = np.array(game_size)/sum(game_size))
	#print(game_files)
	games = [json.load(open(GAME_DIR + game_f)) for game_f in game_files]
	game_pos = [(g, np.random.randint(len(g['history']))) for g in games]
	pos = np.array([[make_input(g['history'], i, 1), *make_target(g['winner'], g['probabilities'], i)] 
		for (g, i) in game_pos
		])

	return list(pos[:,0]), list(pos[:,1]), list(pos[:,2])
 

def train_az(model_loc, 
	folder, n_iter = -1,
	n_epochs = -1, 
	lr = 0.01,
	wd = 0.001,
	momentum = 0.9):
	
	scheduler_params = {
		 "milestones": [5000, 10000, 20000],
		 "gamma": 0.1 
		}

	nn = NetWrapper()
	nn.load_traced_model(model_loc)
	nn.build_optim(
		lr = lr, 
		wd = wd, 
		momentum = momentum,
		scheduler_params = scheduler_params)

	i = 0
	loss, v_loss, p_loss = 0, 0, 0
	losses = {
		'epoch': [],
		'loss': [],
		'v_loss': [],
		'p_loss': []
	}
	total_loss = 0
	while True:
		batch = sample_batch(BATCH_SIZE)
		n_loss, nv_loss, np_loss = nn.train(batch)
		loss += n_loss
		v_loss += nv_loss
		p_loss += np_loss
		total_loss += n_loss
		
		
		if i % SAVE_CHECKPOINT == 0 and i != 0:
			print("New model saved")
			nn.save_traced_model(folder = folder, model_name = "traced_model_new.pt")
		
		if i % TEST_SAVE_CHECKPOINT == 0:
			nn.save_traced_model(folder = folder, model_name = "{}_traced_model_new.pt".format(i))
			df = pd.DataFrame(losses)
			df.to_csv("temp/losses_{}.csv".format(n_iter),index=False) 
		
		if i % LOSS_LOG == 0:
			time.sleep(0.1)
			print("Bacth: {}, \
				loss: {}, \
				value_loss: {},\
				policy_loss: {}".format(i, 
								loss/LOSS_LOG,  
								v_loss/LOSS_LOG, 
								 p_loss/LOSS_LOG))
			
			losses['epoch'].append(i)
			losses['loss'].append(loss/LOSS_LOG)
			losses['v_loss'].append(v_loss/LOSS_LOG)
			losses['p_loss'].append(p_loss/LOSS_LOG)
			loss, v_loss, p_loss = 0, 0, 0

		if n_epochs > 0 and i > n_epochs:
			print(n_iter)
			nn.save_traced_model(folder = folder, model_name = "traced_model_new.pt")
			nn.save_traced_model(folder = folder, model_name = "{}_traced_model_new.pt".format(n_iter))
		

			break

		i += 1


	return total_loss/i

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Train network')
	parser.add_argument('--model', help='model loc')
	parser.add_argument('--folder', help='where to save the new models')
	parser.add_argument('--n_iters',  type=int, help='n iters to run')
	args = parser.parse_args()

	train_az(args.model, args.folder, n_iter = args.n_iters)
