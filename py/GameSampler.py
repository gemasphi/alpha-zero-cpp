import os
import numpy as np
import sys
import json

class Sampler:
	def __init__(self, game_dir, game_window, input_planes, batch_size):
		self.game_dir = game_dir
		self.game_window = game_window
		self.input_planes = input_planes
		self.batch_size = batch_size
		self.n_batches = 0

	def make_input(self, history, i, t):
		player = 1 if i % 2 == 0 else -1
		history = np.array(history)
		size = (t, *(history[0].shape))
		history = np.append(np.zeros(size), history, axis = 0)
		return player*history[i: i + t] 

	def make_target(self, winner, probs, i):
		player = 1 if i % 2 == 0 else -1  
		return probs[i], winner*player 

	def sample_batch(self):
		while True:
			n_games = len(os.listdir(self.game_dir)) 
			all_games = sorted(
				os.listdir(self.game_dir), 
				key = lambda f: os.path.getctime("{}/{}".format(self.game_dir, f))
				)[n_games - round(n_games*self.game_window):]

			game_size = [os.stat(self.game_dir + file).st_size for file in all_games]
			game_files = np.random.choice(all_games, size = self.batch_size,  p = np.array(game_size)/sum(game_size))
			games = []	
			for game_f in all_games:
				try: 
					games.append(json.load(open(self.game_dir + game_f)))
				except ValueError as e:
					print("Game {} ignored".format(self.game_dir + game_f))
			game_pos = [(g, np.random.randint(len(g['history']))) for g in games]
			pos = np.array([[self.make_input(g['history'], i, self.input_planes), *self.make_target(g['winner'], g['probabilities'], i)] 
				for (g, i) in game_pos
				])

			self.n_batches += 1
			yield (self.n_batches, (list(pos[:,0]), list(pos[:,1]), list(pos[:,2])))
 

class SurpervisedSampler(Sampler):
	def make_target(self, value, probs, i):
		return probs[i], value[i] 

class WholeDatasetSupervised(SurpervisedSampler):
	def __init__(self, game_dir, game_window, input_planes, batch_size):
		super(SurpervisedSampler, self).__init__(game_dir, game_window, input_planes, batch_size)
		self.data = self.process()
		print(self.data.shape)

	def process(self):
		filename = self.game_dir + 'positions.npy'
		if os.path.isfile(filename):
			return np.load(filename, allow_pickle=True)

		n_games = len(os.listdir(self.game_dir)) 
		all_games = sorted(
			os.listdir(self.game_dir), 
			key = lambda f: os.path.getctime("{}/{}".format(self.game_dir, f))
			)[n_games - round(n_games*self.game_window):]

		games = [json.load(open(self.game_dir + game_f)) for game_f in all_games]	
		pos = []

		for g in games:
			for i in range(len(g['history'])):
				pos.append([self.make_input(g['history'], i, self.input_planes), *self.make_target(g['value'], g['probabilities'], i)])

		pos = np.array(pos)
		np.random.shuffle(pos)

		np.save(filename, pos) 
		return pos	

	def sample_batch(self):
		i = 0
		while i < len(self.data):
			until = i + self.batch_size if i + self.batch_size < len(self.data) else len(self.data)
			batch = self.data[i : until]
			i+= self.batch_size
			self.n_batches += 1
			yield (self.n_batches, (list(batch[:,0]), list(batch[:,1]), list(batch[:,2])))

class WholeDataset(Sampler):
	def __init__(self, game_dir, game_window, input_planes, batch_size):
		super().__init__(game_dir, game_window, input_planes, batch_size)
		self.data = self.process()
		print(self.data.shape)

	def process(self):
		n_games = len(os.listdir(self.game_dir)) 
		all_games = sorted(
			os.listdir(self.game_dir), 
			key = lambda f: os.path.getctime("{}/{}".format(self.game_dir, f))
			)[n_games - round(n_games*self.game_window):]


		games = []	
		for game_f in all_games:
			try: 
				games.append(json.load(open(self.game_dir + game_f)))
			except ValueError as e:
				print("Game {} ignored".format(self.game_dir + game_f))

		pos = []
		for g in games:
			for i in range(len(g['history'])):
				pos.append([self.make_input(g['history'], i, self.input_planes), *self.make_target(g['winner'], g['probabilities'], i)])

		pos = np.array(pos)
		np.random.shuffle(pos)

		#np.save(filename, pos) 
		return pos	

	def sample_batch(self):
		i = 0
		while i < len(self.data):
			until = i + self.batch_size if i + self.batch_size < len(self.data) else len(self.data)
			batch = self.data[i : until]
			i+= self.batch_size
			self.n_batches += 1
			yield (self.n_batches, (list(batch[:,0]), list(batch[:,1]), list(batch[:,2])))
