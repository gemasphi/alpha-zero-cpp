import numpy as np
import json 
import os
import collections
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import subprocess

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


def elo_plot(file, save_to):
	elo = pd.read_csv(file, delimiter=r"\s+")
	elo = elo.sort_values(by=['Name'], ascending=True)	
	ax = sns.catplot(x="Name", y="Elo", kind="point", data=elo)
	ax.set(xlabel='Generation', ylabel='Elo', title= "Elo progression by generation with MCTS")
	plt.gcf().set_size_inches(8, 5)
	plt.savefig(save_to)

def agreement_plot(file, save_to):
	agree = pd.read_csv(file)
	data = {'Generation': agree.index.values*5, 'Move agreement %': agree[agree.columns[1]].values} 
	data = pd.DataFrame(data)
	ax = sns.catplot(x= "Generation", y="Move agreement %", kind="point", data=data)
	ax.set(title= "Move agreement % with the perfect player by generation")
	plt.gcf().set_size_inches(8, 5)
	plt.savefig(save_to)


def build_pgn(j,i,p1,p2, result):
	return  '[Event "dGrand match" ]\n[Iteration "{}"]\n[Site "your house"]\n[Round "{}"]\n[White "{}"]\n[Black "{}"]\n[Result "{}"]\n random text\n\n'.format(j, i, p1, p2, result)

def matches_to_pgn(dir, file = "matches.pgn"):
	all_matches = sorted(
		os.listdir(dir), 
		key = lambda f: os.path.getctime("{}/{}".format(dir, f))
		)

	with open(file, "w") as f:
		j = 0
		for match_f in all_matches:
			match = json.load(open(dir + match_f))[0]
			
			i = 0
			for result in match['results']:
				if result['winner'] == -1:
					res_str = '0-1'
				elif result['winner'] == 1:
					res_str = '1-0'
				else:
					res_str = '1/2-1/2'	
				
				s = build_pgn(j, i, result['p1'], result['p2'], res_str)
				f.write(s)
				i+=1

			j+=1


def matches_to_agreement(dir, file="agreement.csv"):
	all_matches = sorted(
		os.listdir(dir), 
		key = lambda f: os.path.getctime("{}/{}".format(dir, f))
		)

	agreement = {
		"player": [],
		"agreement": []
	}

	for match_f in all_matches:
		match = json.load(open(dir + match_f))[0]
		agreement["player"].append(match["p1"]["name"])
		agreement["agreement"].append(match["p1"]["move_agreement"])
	
	df = pd.DataFrame(agreement)
	df.to_csv(file, index=False) 




