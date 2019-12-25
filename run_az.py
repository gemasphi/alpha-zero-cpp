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
	result = subprocess.run(['cc/build/game_info', game], stdout= subprocess.PIPE)
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
		jobs.append(subprocess.Popen(['cc/build/selfplay', game, model_loc, str(n_games)],  stdout = log_to))

	return jobs

def launch_training_job(model_loc, n_iter = -1, n_epochs = -1, log_to = ""):
	#ctx = multiprocessing.get_context("spawn")
	#p = ctx.Process(target=train_az, args=(net, "models", n_iter, n_epochs))
	return subprocess.Popen(['python','train.py', model_loc, "models", str(n_iter), str(n_epochs)],  stdout = log_to)


def launch_play_agaisnt_job(game, model_loc, n_games = -1, log_to = ""):
	return subprocess.Popen(['cc/build/play_agaisnt', game, model_loc, str(n_games)], stdout = log_to)

GAME = "CONNECTFOUR"
N_JOBS = 1
AYNSC = False
N_ITERS = 500 
N_EPOCHS = 1500
TEST_INTERVAL = 15

train_log, selfplay_log, play_agaisnt_log = setup_logs()

if AYNSC:
	cpu_loc, gpu_loc, net = build_network(GAME)
	jobs = launch_selfplay_jobs(GAME, cpu_loc, n_jobs = N_JOBS , log_to = selfplay_log)
	p = launch_training_job(gpu_loc, log_to = train_log)

	#while True:
	#	launch_play_agaisnt_job(GAME, cpu_loc, n_games = 100, log_to = play_agaisnt_log).wait()	
	#	time.sleep(TEST_INTERVAL * 60)

	p.wait()
	for j in jobs:
		j.wait()
else:
	cpu_loc, gpu_loc = "models/cpu_traced_model_new.pt", "models/gpu_traced_model_new.pt"
	#cpu_loc, gpu_loc, net = build_network(GAME)
	for i in range(N_ITERS):
		jobs = launch_selfplay_jobs(GAME, cpu_loc, n_games = 40, n_jobs = N_JOBS,  log_to = selfplay_log)
		
		for j in jobs:
			j.wait()
		
		print("Training started")
		p = launch_training_job(gpu_loc, i, N_EPOCHS,  log_to = train_log)
		p.wait()
		print("Training ended")

		#launch_play_agaisnt_job(GAME, model_loc, n_games = 10).wait()
