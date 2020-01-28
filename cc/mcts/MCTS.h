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


class MCTS
{
	private:
		float cpuct;
		float dirichlet_alpha;

	public:
		MCTS(float cpuct, float dirichlet_alpha);
		ArrayXf simulate(std::shared_ptr<Game> game, NNWrapper& model, float temp = 1, int n_simulations = 25);
};

#endif 
