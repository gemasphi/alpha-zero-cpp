import subprocess
import json 
from py.NN import NetWrapper 
from train import train_az
from multiprocessing import Process
import time


def setup_logs(folder = 'temp'):
	if not os.path.isdir(folder):
		os.mkdir(folder)

	train_log = open('training.log', 'a')
	selfplay_log = open('selfplay.log', 'a')
	play_agaisnt_log = open('play_agaisnt.log', 'a')

	return train_log, selfplay_log, play_agaisnt_log

def build_network(game):
	result = subprocess.run(['cc/build/game_info', game], stdout= subprocess.PIPE)
	game_info = json.loads(result.stdout.decode('utf-8'))

	net = NetWrapper(
		input_planes = game_info["input_planes"], 
		board_dim = game_info["board_size"], 
		output_planes = game_info["output_planes"], 
		action_size = game_info["action_size"], 
		res_layer_number = 5
		)

	folder = "models"
	model_name = 'traced_model_new.pt'
	model_loc = "{}/{}".format(folder, model_name)
	net.save_traced_model(folder = folder, model_name = model_name)

	return model_loc, net

def launch_selfplay_jobs(game, model_loc, n_games = -1, n_jobs = 2, log_to = ""):
	jobs = []
	for i in range(n_jobs):
		print("Starting selfplay job {}/{}".format(i + 1, n_jobs))
		jobs.append(subprocess.Popen(['cc/build/selfplay', game, model_loc, str(n_games)]))

	return jobs

def launch_training_job(net, n_iter = -1, n_epochs = -1):
	p = Process(target=train_az, args=(net, "models", n_iter, n_epochs))
	p.start()
	return p

def launch_play_agaisnt_job(game, model_loc, n_games = -1):
	return subprocess.Popen(['cc/build/play_agaisnt', game, model_loc, str(n_games)])

GAME = "CONNECTFOUR"
N_JOBS = 1
AYNSC = False
N_ITERS = 50 
N_EPOCHS = 350

if AYNSC:
	model_loc, net = build_network(GAME)
	jobs = launch_selfplay_jobs(GAME, model_loc, n_jobs = N_JOBS)
	time.sleep(60)
	p = launch_training_job(net)
	p.join()
	for j in jobs:
		j.wait()
else:
	model_loc, net = build_network(GAME)
	for i in range(N_ITERS):
		jobs = launch_selfplay_jobs(GAME, model_loc, n_games = 15, n_jobs = N_JOBS)
		
		for j in jobs:
			j.wait()

		p = launch_training_job(net, i, N_EPOCHS)
		p.join()
		print("Training ended")

		#launch_play_agaisnt_job(GAME, model_loc, n_games = 10).wait()
