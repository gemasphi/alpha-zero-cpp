#ifndef MCTS_H_
#define MCTS_H_

#include <iostream>
#include <map>
#include <NNWrapper.h>
#include <Game.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <memory>
#include <limits> 
#include "GameState.h"

using namespace Eigen;


namespace MCTS
{
	struct Config
	{
		float cpuct = 1;
		float dirichlet_alpha = 1;
		float n_simulations = 25;
		float temp = 1;
	};

	ArrayXf simulate(std::shared_ptr<Game> game, NNWrapper& model, MCTS::Config cfg);
	ArrayXf simulate(std::shared_ptr<GameState> root, NNWrapper& model, MCTS::Config cfg);
	ArrayXf parallel_simulate(std::shared_ptr<GameState> root, NNWrapper& model, MCTS::Config cfg);
}

#endif 
