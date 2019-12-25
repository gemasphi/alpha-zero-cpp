import numpy as np
import json 
import os
import collections

GAME_DIR = "games/"

def count_last_positions(dir):
	pos = set() 
	total_pos = []
	for f in os.listdir(dir):
		game = json.load(open(dir + f))
		
		for p in game['history']:
			pos.add(str(p))
			total_pos.append(str(p))

	print(len(total_pos))
	print(len(pos))
count_last_positions(GAME_DIR)
