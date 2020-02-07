import subprocess
import json
from py.NN import NetWrapper 
from multiprocessing import Process
import multiprocessing
import time
import torch
import os

def setup_logs(folder = 'temp'):
	if not os.path.isdir(folder):
		os.mkdir(folder)

	train_log = open('{}/training.log'.format(folder), 'a')
	selfplay_log = open('{}/selfplay.log'.format(folder), 'a')
	play_agaisnt_log = open('{}/play_agaisnt.log'.format(folder), 'a')

	return train_log, selfplay_log, play_agaisnt_log

def build_network(game):
	result = subprocess.run(['build/game_info', game], stdout= subprocess.PIPE)
	game_info = json.loads(result.stdout.decode('utf-8'))

	net = NetWrapper()
	net.build(
		input_planes = game_info["input_planes"], 
		board_dim = game_info["board_size"], 
		output_planes = game_info["output_planes"], 
		action_size = game_info["action_size"], 
		res_layer_number = 10
		)

	folder = "models"
	model_name = 'traced_model_new.pt'
	cpu_loc, gpu_loc = net.save_traced_model(folder = folder, model_name = model_name)

	return cpu_loc, gpu_loc, net


if __name__ == "__main__":
	GAME = "CONNECTFOUR"
	SAVE_MODELS = "models/"
	N_GAMES = 100
	N_ITERS = 500
	N_GENS = 20

	train_log, selfplay_log, play_agaisnt_log = setup_logs()
	cpu_loc, gpu_loc, net = build_network(GAME)

	for i in range(N_GENS):
		print("Starting Selfplay")
		subprocess.Popen(['build/selfplay', 
						'--game={}'.format(GAME), 
						'--model={}'.format(cpu_loc),
						'--n_games={}'.format(N_GAMES),
						],  stdout = selfplay_log).wait()

		subprocess.Popen(['python3','train.py', 
						'--model={}'.format(cpu_loc),
						'--folder={}'.format(SAVE_MODELS),
						'--n_iters={}'.format(N_ITERS),
						],  stdout = train_log).wait()