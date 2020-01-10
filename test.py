import subprocess
from utils import matches_stats
import numpy as np
def launch_play_agaisnt_job(game, p1 = "", n_games = -1, perfect_player_loc = "", p2 = "", id = -1):
	print("{} vs {}".format( p1,p2))
	return subprocess.Popen(['cc/build/play_agaisnt', game, p1, str(n_games), perfect_player_loc, p2, str(id)])

def model_loc(i):
	return "models/gpu_{}_traced_model_new.pt".format(i)
 

def build_pgn(j,i,p1,p2, result):
	return  '[Event "dGrand match" ]\n[Iteration "{}"]\n[Site "your house"]\n[Round "{}"]\n[White "{}"]\n[Black "{}"]\n[Result "{}"]\n random text\n\n'.format(j, i, p1, p2, result)

#oldvsnew 
def matches_stats_to_pgn(matches_stats, file = "matches.pgn"):
	"""
	print(matches_stats)
	matches_stats["losses"], matches_stats["wins"] = np.where(matches_stats.index % 2 == 0, [matches_stats["wins"], matches_stats["losses"]], [matches_stats["losses"], matches_stats["wins"] ])
	print(matches_stats)

	d = {'draws': 'sum', 'losses': 'sum', 'wins':'sum'}
	matches_stats = matches_stats.groupby(matches_stats.index // 2).agg(d)
	print(matches_stats)
	"""

	
	with open(file, "a") as f:
		j = 0
		for m in range(len(matches)):
			
			results = []
			results += ['1/2-1/2'for _ in range(match['draws'])]
			results += ['1-0'for _ in range(match['wins'])]
			results += ['0-1'for _ in range(match['losses'])]
			i = 0
			
			for res in results:	
				s = build_pgn(j, i, j + 1, j, res)
				f.write(s)
				i+=1

			j+=1


def matches_to_pgn(matches, matches_stats, file = "matches.pgn"):
	with open(file, "a") as f:
		j = 0
		for match in matches:
			print(matches)
			results = []
			id = match['id']

			results += ['1/2-1/2'for _ in range(matches_stats['draws'][id])]
			results += ['1-0'for _ in range(matches_stats['wins'][id])]
			results += ['0-1'for _ in range(matches_stats['losses'][id])]
			
			i = 0
			for res in results:	
				s = build_pgn(j, i, match['p1'], match['p2'], res)
				f.write(s)
				i+=1
			j+=1

N_MODELS = 50
MODEL_STEP = 5
GAME = "CONNECTFOUR"
"""
matches = []
id = 0

matches = [{"id": 0, "p1": 0, "p2": 5 },
{"id": 1, "p1": 5, "p2": 0 },
{"id": 2, "p1": 5, "p2": 10 },
{"id": 3, "p1": 10, "p2": 5 },
{"id": 4, "p1": 10, "p2": 15 },
{"id": 5, "p1": 15, "p2": 10 },
{"id": 6, "p1": 15, "p2": 20 },
{"id": 7, "p1": 20, "p2": 15 },
{"id": 8, "p1": 0, "p2": -1 }]

matches_to_pgn(matches, matches_stats("temp/playagaisnt_games/"))

for i in range(MODEL_STEP, N_MODELS, MODEL_STEP):
	p1 = i - MODEL_STEP
	p2 = i

	launch_play_agaisnt_job(GAME, p1 = model_loc(p1), n_games = 50, p2 = model_loc(p2), id = id).wait()
	matches.append({"id": id, "p1": p1, "p2": p2 })
	id += 1

	launch_play_agaisnt_job(GAME, p1 = model_loc(p2), n_games = 50, p2 = model_loc(p1), id = id).wait()
	matches.append({"id": id, "p1": p2, "p2": p1 })
	id += 1

np.savetxt('matches.csv', np.asarray(matches), delimiter=',')
"""