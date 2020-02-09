import subprocess
import numpy as np
import os 
import json
from utils import *
import time
from multiprocessing import Process

def bayeselo(pgn_file, to):
	bayes = subprocess.Popen(['build/bayeselo-prefix/src/bayeselo/bayeselo'],   
								stdin=subprocess.PIPE, shell=True)

	bayes.stdin.write('readpgn {}\n'.format(pgn_file).encode())
	bayes.stdin.write('elo\n'.encode())
	bayes.stdin.write('mm\n'.encode())
	bayes.stdin.write('exactdist\n'.encode())
	bayes.stdin.write('ratings >{}\n'.format(to).encode())
	bayes.stdin.write('x\n'.format(to).encode())
	bayes.stdin.write('x\n'.format(to).encode())
	bayes.stdin.close()
	time.sleep(1) #litte hacky but needed

if __name__ == "__main__":
	SAVE_DIRECTORY = "temp/test/"
	VS_DIR = "temp/playagaisnt_games/vs/"
	PGN = SAVE_DIRECTORY + "vs.pgn"
	ELO = SAVE_DIRECTORY + "elo.txt"
	ELO_PLOT = SAVE_DIRECTORY + "elo.png"

	if not os.path.isdir(SAVE_DIRECTORY):
		os.mkdir(SAVE_DIRECTORY)

	matches_to_pgn(VS_DIR, PGN)
	bayeselo(PGN, ELO)
	elo_plot(ELO, ELO_PLOT)

	AGREEMENT_DIR = "temp/playagaisnt_games/agreement/"
	AGREEMENT_CSV = SAVE_DIRECTORY + "agreement.csv"
	AGREEMENT_PLOT = SAVE_DIRECTORY + "agreement.png"

	matches_to_agreement(AGREEMENT_DIR, AGREEMENT_CSV)
	agreement_plot(AGREEMENT_CSV, AGREEMENT_PLOT)
