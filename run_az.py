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

def launch_selfplay_jobs(game, model_loc, n_games = -1, n_jobs = 2, log_to = ""):
	jobs = []
	for i in range(n_jobs):
		print("Starting selfplay job {}/{}".format(i + 1, n_jobs))
		jobs.append(subprocess.Popen(['build/selfplay', game, model_loc, str(n_games)],  stdout = log_to))

	return jobs

def launch_training_job(model_loc, n_iter = -1, n_epochs = -1, log_to = ""):
	#ctx = multiprocessing.get_context("spawn")
	#p = ctx.Process(target=train_az, args=(net, "models", n_iter, n_epochs))
	return subprocess.Popen(['python','train.py', model_loc, "models", str(n_iter), str(n_epochs)],  stdout = log_to)


def launch_play_agaisnt_job(game, model_loc, n_games = -1, perfect_player_loc = "", p2 = ""):
	return subprocess.Popen(['build/play_agaisnt', game, model_loc, str(n_games), perfect_player_loc, p2])

GAME = "CONNECTFOUR"
N_JOBS = 1
AYNSC = True
N_ITERS = 500
N_EPOCHS = 10000
TEST_INTERVAL = 15

train_log, selfplay_log, play_agaisnt_log = setup_logs()

if AYNSC:
	i = 0
	cpu_loc, gpu_loc, net = build_network(GAME)
	jobs = launch_selfplay_jobs(GAME, cpu_loc, n_jobs = N_JOBS , log_to = selfplay_log)

	print("Launched")
	time.sleep(2*60)
	print("Training started")
	p = launch_training_job(gpu_loc,i, -1, log_to = train_log)
	p.wait()
	print("Testing started")
	#launch_play_agaisnt_job(GAME, gpu_loc, n_games = 25)
	i += 1
	"""
	for j in jobs:
		j.wait()
	"""
else:
	#cpu_loc, gpu_loc = "models/cpu_traced_model_new.pt", "models/gpu_traced_model_new.pt"

	#cpu_loc, gpu_loc, net = build_network(GAME)
	#cpu_loc = "models/cpu_traced_model_new.pt"
	#gpu_loc = "models/gpu_traced_model_new.pt"
	for i in range(N_ITERS):

		jobs = launch_selfplay_jobs(GAME, cpu_loc, n_games = 30, n_jobs = N_JOBS,  log_to = selfplay_log)

		for j in jobs:
			j.wait()

		print("Training started")
		p = launch_training_job(gpu_loc, i, N_EPOCHS,  log_to = train_log)
		print("Training ended")
		p.wait()
		

		#launch_play_agaisnt_job(GAME, "models/gpu_{}_traced_model_new.pt".format(i), n_games = 20, p2 = "models/gpu_{}_traced_model_new.pt".format(i - 1))
