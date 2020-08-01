import yaml
from py.GameSampler import Sampler, WholeDatasetSupervised, SurpervisedSampler, WholeDataset
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

def train_az(args):
	nn = NetWrapper()
	nn.load_traced_model(args.model_loc)
	nn.build_optim(
		lr = args.lr, 
		wd = args.wd, 
		momentum = args.momentum)

	sampler = Sampler(
				args.data_location, 
				args.n_games, 
				args.input_planes,
				args.train_batchsize)

	
	full_stats = []
	stats = Stats()
	complete_stats = Stats()

	for i, batch in sampler.sample_batch():
		stats += nn.train(batch)
		if i != 0 and i % loss_log == 0:
			complete_stats += stats
			stats.log(i, loss_log)
			full_stats.append(stats)
			stats = Stats()

		if args.n_iter > 0 and i > args.n_iter:
			break

		if i != 0 and i % 250 == 0:
			if not os.path.isdir('temp/losses/'):
				os.mkdir('temp/losses/')
			
			df = pd.DataFrame([asdict(s) for s in full_stats])
			df.to_csv("temp/losses/{}_losses.csv".format(args.current_gen),index=False) 
					
			nn.save_traced_model(folder = folder, model_name = "traced_model_new.pt")
			nn.save_traced_model(folder = folder, model_name = "{}_traced_model_new.pt".format(i))

	return complete_stats.loss

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Train network')
	parser.add_argument('--model_loc', help='model loc')
	parser.add_argument('--save_folder', help='where to save the new models')
	parser.add_argument('--n_iter',  type=int, help='n iters to run')
	parser.add_argument('--current_gen',  type=int, help='current generation')
	parser.add_argument('--loss_log',  type=int, help='log loss every n iterations')
	parser.add_argument('--train_batchsize',  type=int, help='log loss every n iterations')
	parser.add_argument('--lr', type=float)
	parser.add_argument('--wd', type=float)
	parser.add_argument('--momentum', type=float)
	parser.add_argument('--data_location', type=str)
	parser.add_argument('--n_games', type=int)
	parser.add_argument('--input_planes', type=int)


	args = parser.parse_args()
	train_az(args)
