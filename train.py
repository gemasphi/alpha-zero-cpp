import yaml
from py.GameSampler import Sampler, SurpervisedSampler
from py.NN import NetWrapper, Stats
import json 
import os
import numpy as np
import torch.optim as optim
#import subprocess
import sys
import time
import pandas as pd
import argparse
from dataclasses import asdict


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

	sampler = SurpervisedSampler(
				data['location'], 
				data['n_games'], 
				nn_params['input_planes'],
				nn_params['batch_size'])

	
	full_stats = []
	stats = Stats()
	i = 0
	while True:
		batch = sampler.sample_batch()
		stats += nn.train(batch)

		if i != 0 and i % loss_log == 0:
			stats.log(i, loss_log)
			full_stats.append(stats)
			stats = Stats()

		if n_iter > 0 and i > n_iter:
			if not os.path.isdir('temp/losses/'):
				os.mkdir('temp/losses/')
			df = pd.DataFrame([asdict(s) for s in full_stats])
			df.to_csv("temp/losses/{}_losses.csv".format(n_gen),index=False) 
			
			nn.save_traced_model(folder = folder, model_name = "traced_model_new.pt")
			nn.save_traced_model(folder = folder, model_name = "{}_traced_model_new.pt".format(n_gen))

			break

		i += 1


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
	train_az(args.model, 
			args.folder, 
			args.nn_params,
			args.data,
			loss_log = args.loss_log, 
			n_iter = args.n_iter, 
			n_gen = args.n_gen)
