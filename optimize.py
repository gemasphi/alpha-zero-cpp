import yaml
import optuna
from py.NN import NetWrapper
from train import train_az

"{'lr': 0.010942571921826386, 'wd': 0.0012998839635123788, 'momentum': 0.002209938938990752}."
def objective(trial):
	lr = trial.suggest_loguniform('lr', 0.005, 0.5)
	wd = trial.suggest_loguniform('wd', 0.001, 0.05)
	momentum = trial.suggest_loguniform('momentum', 0.001, 0.05)
	
	NN_PARAMS = {
		"input_planes": 1,
		"batch_size": 1024,
		"lr" : lr,
		"wd" : wd,
		"momentum" : momentum,
		"scheduler_params" : {}
	}

	DATA = {
		"location": "build/temp/perfect_player/",
		"n_games": 0.5
	}
	return train_az(
		"temp/models/traced_model_new.pt",
		"temp/models/", 
		NN_PARAMS, 
		DATA)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials = 100)
print('Number of finished trials: ', len(study.trials))
print('Best trial:')
trial = study.best_trial
print('  Value: ', trial.value)