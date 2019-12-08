import yaml
import optuna
from src.AlphaZeroTrainer import AlphaZeroTrainer as az 
from src.NN import NetWrapper, trace_model
from src.games.Tictactoe import Tictactoe
from src.Player import * 
from src.MCTS_virtual_loss import MCTS

#{'lr': 0.012486799182525229, 'wd': 0.015010724465999522}.

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

game = Tictactoe(**config['GAME'])

trace_model(game)

"""
mcts = MCTS(**config['MCTS'])
nn = NetWrapper(game, **config['NN'])
#nn.load_model()

alphat = az(nn, game, mcts, **config['AZ'])
loss = alphat.train(lr = 0.01, wd = 0.015)
"""