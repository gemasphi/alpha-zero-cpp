import numpy as np
import json 
import os
import collections
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

GAME_DIR = "temp/games/"
MATCHES_DIR = "temp/playagaisnt_games/"

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

def calc_agreement(match, who):
	n_random_moves = 0
	total_agreement = 0 
	total_agreement_sum = 0
	for result in match["results"]:
		agreement = result[who][n_random_moves:]
		total_agreement +=  np.count_nonzero(agreement) 
		total_agreement_sum += len(agreement)

	if total_agreement == 0:
		return 0

	return total_agreement/total_agreement_sum

def matches_stats(dir):
	stats = {
		'draws': [],
		'wins': [],
		'losses': [],
		'agreement_p1': [],
		'agreement_p2': []
	}
	matches = sorted(
		os.listdir(dir), 
		key = lambda f: os.path.getctime("{}/{}".format(dir, f))
		)
	for f in matches:
		match = json.load(open(dir + f))
		stats['wins'].append(match['p1_wins'])
		stats['losses'].append(match['p2_wins'])
		stats['draws'].append(match['draws']) 
		stats['agreement_p1'].append(calc_agreement(match,'agreement1'))
		stats['agreement_p2'].append(calc_agreement(match,'agreement2'))
			
	df = pd.DataFrame(stats)
	return df

def elo_plot(file = 'elo'):
	elo = pd.read_csv(file, delimiter=r"\s+")
	elo = elo.sort_values(by=['Name'], ascending=True)	
	print(elo)
	#elo = elo[elo['Name'] != -1]
	elo.loc[elo['Name']== -1, 'Name'] = "Random" 
	elo.loc[elo['Name']== 0, 'Name'] = 1 
	print(elo)
	ax = sns.catplot(x="Name", y="Elo", kind="point", data=elo)
	ax.set(xlabel='Generation', ylabel='Elo', title= "Elo progression by generation with MCTS")
	#ax.set(ylim=(-400, 400))
	plt.gcf().set_size_inches(8, 5)
	plt.savefig('elo.png')

def agreement_plot(file="temp/agreement.cvs"):
	agree = pd.read_csv(file, index_col=0)
	data = {'Generation': agree.index.values*5, 'Move agreement %': agree[agree.columns[0]].values} 
	data = pd.DataFrame(data)
	data.loc[data['Generation']== 0, 'Generation'] = 1 
	ax = sns.catplot(x= "Generation", y="Move agreement %", kind="point", data=data)
	ax.set(title= "Move agreement % with the perfect player by generation")
	plt.gcf().set_size_inches(8, 5)
	plt.savefig('agreement.png')


elo_plot()
agreement_plot()

#count_last_positions(GAME_DIR)
#df = matches_stats(MATCHES_DIR)
#print(df)
#df.to_csv("temp/agreement.cvs",index=True)
