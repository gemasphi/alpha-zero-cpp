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
		res_layer_number = 12
		)

	net.save_traced_model(folder = folder, model_name = '-1_traced_model_new.pt')
	model_loc = net.save_traced_model(folder = folder, model_name = 'traced_model_new.pt')
	return model_loc

#"{'lr': 0.010942571921826386, 'wd': 0.0012998839635123788, 'momentum': 0.002209938938990752}."

if __name__ == "__main__":
	GAME = "CONNECTFOUR"
	N_GENS = 100
	N_SELFPLAY_GAMES = 100000000
	N_PLAYAGAISNT_GAMES = 100

	N_ITERS = -1
	SAVE_MODELS = "temp/models/"
	LOSS_LOG = 5

	NN_PARAMS = {
		"batch_size": 1024,
		"lr" : 0.001,
		"wd" : 0.0005,
		"momentum" : 0.9,
		"scheduler_params" : {}
	}

	DATA = {
		"location": "temp/games/",
		"n_games": 1500
	}
        
	train_log, selfplay_log, play_agaisnt_log = setup_logs()
	#model_loc = build_network(GAME, SAVE_MODELS, NN_PARAMS)
	model_loc = "temp/models/traced_model_new.pt"
	NN_PARAMS['input_planes'] = 1
	
	for i in range(1, N_GENS):
		print("Starting Selfplay")
		subprocess.Popen(['build/selfplay', 
						'--game={}'.format(GAME), 
						'--model={}'.format(model_loc),
						'--n_games={}'.format(N_SELFPLAY_GAMES),
						'--cpuct={}'.format(2.5),
						'--n_simulations={}'.format(800),
						'--mcts_threads={}'.format(1),
						'--selfplay_threads={}'.format(31),
						'--global_batch_size={}'.format(31),
						'--batch_size={}'.format(1),
						], stdout = selfplay_log)
		
		print("Started Training")
		subprocess.Popen(['python3','train.py', 
						'--model={}'.format(model_loc),
						'--folder={}'.format(SAVE_MODELS),
						'--n_iter={}'.format(N_ITERS),
						'--n_gen={}'.format(i),
						'--loss_log={}'.format(LOSS_LOG),
						'--nn_params={}'.format(json.dumps(NN_PARAMS)),
						'--data={}'.format(json.dumps(DATA)),
						], stdout= train_log).wait()
		#print("{} iters trained: {}".format(N_ITERS, time.time() - start_time))
		
		
		"""
		print("Started Play Agaisnt Match")
		start_time = time.time()
		subprocess.Popen(['build/play_agaisnt', 
						'--id={}'.format(i),						
						'--n_games={}'.format(N_PLAYAGAISNT_GAMES),						
						'--game={}'.format(GAME),						
						'--model_one=temp/models/{}_traced_model_new.pt'.format(i),						
						'--model_two=temp/models/{}_traced_model_new.pt'.format(i - 1),						
						], stdout = play_agaisnt_log).wait()


		print("{} games played: {}".format(N_PLAYAGAISNT_GAMES*2, time.time() - start_time))
		"""
