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

def make_input(history, i, t):
	player = -1 if i % 2 == 0 else 1
	history = np.array(history)
	size = (t, *(history[0].shape))
	history = np.append(np.zeros(size), history, axis = 0)
	return player*history[i: i + t] 

def make_target(winner, probs, i):
	player = -1 if i % 2 == 0 else 1  
	return probs[i], winner*player 

def sample_batch(game_dir, game_window, input_planes, batch_size):
	n_games = len(os.listdir(game_dir)) 
	all_games = sorted(
		os.listdir(game_dir), 
		key = lambda f: os.path.getctime("{}/{}".format(game_dir, f))
		)[:round(n_games*game_window)]

	game_size = [os.stat(game_dir + file).st_size for file in all_games]
	game_files = np.random.choice(all_games, size = batch_size,  p = np.array(game_size)/sum(game_size))
	games = [json.load(open(game_dir + game_f)) for game_f in game_files]
	game_pos = [(g, np.random.randint(len(g['history']))) for g in games]
	pos = np.array([[make_input(g['history'], i, input_planes), *make_target(g['winner'], g['probabilities'], i)] 
		for (g, i) in game_pos
		])

	return list(pos[:,0]), list(pos[:,1]), list(pos[:,2])
 
def train_az(
	model_loc,	
	folder, 
	nn_params, 
	data,
	n_iter = -1, 
	n_gen = -1,
	loss_log = 20
	):
	nn = NetWrapper()
	nn.load_traced_model(model_loc)
	nn.build_optim(
		lr = nn_params['lr'], 
		wd = nn_params['wd'], 
		momentum = nn_params['momentum'],
		scheduler_params = nn_params['scheduler_params']
	)

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
		batch = sample_batch(
				data['location'], 
				data['n_games'], 
				nn_params['input_planes'],
				nn_params['batch_size'])

		n_loss, nv_loss, np_loss = nn.train(batch)
		loss += n_loss
		v_loss += nv_loss
		p_loss += np_loss
		total_loss += n_loss
		
		if i % loss_log == 0:
			print("Batch: {}, \
				loss: {}, \
				value_loss: {},\
				policy_loss: {}".format(i, 
								loss/loss_log,  
								v_loss/loss_log, 
								 p_loss/loss_log))
			
			losses['epoch'].append(i)
			losses['loss'].append(loss/loss_log)
			losses['v_loss'].append(v_loss/loss_log)
			losses['p_loss'].append(p_loss/loss_log)
			loss, v_loss, p_loss = 0, 0, 0

		if n_iter > 0 and i > n_iter:
			if not os.path.isdir('temp/losses/'):
				os.mkdir('temp/losses/')
			df = pd.DataFrame(losses)
			df.to_csv("temp/losses/{}_losses.csv".format(n_gen),index=False) 
			nn.save_traced_model(folder = folder, model_name = "traced_model_new.pt")
			nn.save_traced_model(folder = folder, model_name = "{}_traced_model_new.pt".format(n_gen))

			break

		i += 1


	return total_loss/i

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Train network')
	parser.add_argument('--model', help='model loc')
	parser.add_argument('--folder', help='where to save the new models')
	parser.add_argument('--n_iter',  type=int, help='n iters to run')
	parser.add_argument('--n_gen',  type=int, help='current generation')
	parser.add_argument('--loss_log',  type=int, help='log loss every n iterations')
	parser.add_argument('--nn_params', type=json.loads)
	parser.add_argument('--data', type=json.loads)
	args = parser.parse_args()
	print(args)
	train_az(args.model, 
			args.folder, 
			args.nn_params,
			args.data,
			loss_log = args.loss_log, 
			n_iter = args.n_iter, 
			n_gen = args.n_gen)
