import yaml
import optuna
from py.NN import NetWrapper
from train import train_az
from params import TrainParams,MCTSParams, to_args, instanciate_params_from_args

def objective(trial):
	lr = trial.suggest_loguniform('lr', 0.005, 0.5)
	wd = trial.suggest_loguniform('wd', 0.001, 0.05)
	momentum = trial.suggest_loguniform('momentum', 0.001, 0.05)
	
	parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
	parser.add_argument('--input_planes', --default=1, type=int)
	parser.add_argument('--current_gen', --default=1, type=int)
	TrainParams.add_to_parser(parser)
	MCTSParams.add_to_parser(parser)
	args = parser.parse_args()
	args.lr = lr
	args.momentum = momentum
	args.wd = wd
	return train_az(args)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials = 100)
print('Number of finished trials: ', len(study.trials))
print('Best trial:')
trial = study.best_trial
print('  Value: ', trial.value)