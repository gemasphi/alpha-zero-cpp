import yaml
import optuna
from py.NN import NetWrapper
from train import train_az

def objective(trial):
	lr = trial.suggest_loguniform('lr', 0.01, 0.5)
	wd = trial.suggest_loguniform('wd', 0.0001, 0.5)
	momentum = trial.suggest_loguniform('mm', 0.01, 1)
	return train_az(
		"models/gpu_traced_model_new.pt",
		"models", 
		-1, 
		500, 
		lr =lr, 
		wd = wd,
		momentum = momentum
		)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials = 100)
print('Number of finished trials: ', len(study.trials))
print('Best trial:')
trial = study.best_trial
print('  Value: ', trial.value)