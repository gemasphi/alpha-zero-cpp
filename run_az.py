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

def build_network(game, folder, nn_params):
	result = subprocess.run(['build/game_info', game], stdout= subprocess.PIPE)
	game_info = json.loads(result.stdout.decode('utf-8'))
	nn_params['input_planes'] = game_info['input_planes']

	net = NetWrapper()
	net.build(
		input_planes = game_info["input_planes"], 
		board_dim = game_info["board_size"], 
		output_planes = game_info["output_planes"], 
		action_size = game_info["action_size"], 
		res_layer_number = 15
		)

	model_loc = net.save_traced_model(folder = folder, model_name = 'traced_model_new.pt')
	model_loc = net.save_traced_model(folder = folder, model_name = '-1_traced_model_new.pt')
	return model_loc


if __name__ == "__main__":
	GAME = "CONNECTFOUR"
	N_GENS = 100
	N_SELFPLAY_GAMES = 7168
	N_PLAYAGAISNT_GAMES = 400

	N_ITERS = 1000
	SAVE_MODELS = "temp/models/"
	LOSS_LOG = 20

	NN_PARAMS = {
		"batch_size": 2048,
		"lr" : 0.01,
		"wd" : 0.001,
		"momentum" : 0.9,
		"scheduler_params" : {
		 "milestones": [250, 500, 750],
		 "gamma": 0.1 
		}
	}
	DATA = {
		"location": "temp/games/",
		"n_games": 500000
	}


	train_log, selfplay_log, play_agaisnt_log = setup_logs()
	model_loc = build_network(GAME, SAVE_MODELS, NN_PARAMS)

	for i in range(N_GENS):
		print("Generaton {}".format(i))
		print("Starting Selfplay")
		subprocess.Popen(['build/selfplay', 
						'--game={}'.format(GAME), 
						'--model={}'.format(model_loc),
						'--n_games={}'.format(N_SELFPLAY_GAMES),
						], stdout = selfplay_log).wait()

		print("Started Training")
		subprocess.Popen(['python3','train.py', 
						'--model={}'.format(model_loc),
						'--folder={}'.format(SAVE_MODELS),
						'--n_iter={}'.format(N_ITERS),
						'--n_gen={}'.format(i),
						'--loss_log={}'.format(LOSS_LOG),
						'--nn_params={}'.format(json.dumps(NN_PARAMS)),
						'--data={}'.format(json.dumps(DATA)),
						], stdout = train_log).wait()

		print("Started Play Agaisnt Match")
		subprocess.Popen(['build/play_agaisnt', 
						'--id={}'.format(i),						
						'--n_games={}'.format(N_PLAYAGAISNT_GAMES),						
						'--game={}'.format(GAME),						
						'--model_one=temp/models/{}_traced_model_new.pt'.format(i),						
						'--model_two=temp/models/{}_traced_model_new.pt'.format(i - 1),						
						], stdout = play_agaisnt_log).wait()
