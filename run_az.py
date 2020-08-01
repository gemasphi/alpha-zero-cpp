import subprocess
import json
from py.NN import NetWrapper 
from multiprocessing import Process
import multiprocessing
import time
import torch
import os
from params import TrainParams,MCTSParams, to_args, instanciate_params_from_args
import argparse

def setup_logs(folder = 'temp'):
	if not os.path.isdir(folder):
		os.mkdir(folder)

	train_log = open('{}/training.log'.format(folder), 'a')
	selfplay_log = open('{}/selfplay.log'.format(folder), 'a')
	play_agaisnt_log = open('{}/play_agaisnt.log'.format(folder), 'a')

	return train_log, selfplay_log, play_agaisnt_log

def build_network(game, folder):
	result = subprocess.run(['build/game_info', game], stdout= subprocess.PIPE)
	game_info = json.loads(result.stdout.decode('utf-8'))

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
	return model_loc, game_info['input_planes']


if __name__ == "__main__":
	parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
	parser.add_argument('--game', help='Game')
	parser.add_argument('--selfplay_games', type=int, help='Number of selfplay games by generation.')
	parser.add_argument('--n_gen', type=int, help='Number of generations')
	parser.add_argument('--testing_games', type=int, help='Number of games used to test by generation.')

	TrainParams.add_to_parser(parser)
	MCTSParams.add_to_parser(parser)
	args = parser.parse_args()
	train_params = instanciate_params_from_args(TrainParams, args)
	mcts_params = instanciate_params_from_args(MCTSParams, args)

	train_log, selfplay_log, play_agaisnt_log = setup_logs()
	model_loc, input_planes = build_network(args.game, train_params.save_folder)
	
	for i in range(1, args.n_gen):
		print("Starting Selfplay")
		selfplay = ['build/selfplay', '--game={}'.format(args.game), '--model={}'.format(model_loc), '--n_games={}'.format(args.selfplay_games)]
		selfplay.extend(to_args(mcts_params))

		subprocess.Popen(selfplay, stdout = selfplay_log)
		
		print("Started Training")
		train = ['python3','train.py', '--model_loc={}'.format(model_loc), "--current_gen={}".format(i), "--input_planes={}".format(input_planes)]
		train.extend(to_args(train_params))
								
		subprocess.Popen(train, stdout= train_log).wait()
		
		
		print("Started Play Agaisnt Match")
		start_time = time.time()
		subprocess.Popen(['build/play_agaisnt', 
						'--id={}'.format(i),						
						'--n_games={}'.format(args.testing_games),						
						'--game={}'.format(args.game),						
						'--model_one=temp/models/{}_traced_model_new.pt'.format(i),						
						'--model_two=temp/models/{}_traced_model_new.pt'.format(i - 1),						
						], stdout = play_agaisnt_log).wait()


		print("{} games played: {}".format(args.testing_games*2, time.time() - start_time))
