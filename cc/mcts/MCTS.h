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
#include <cxxopts.hpp>

using namespace Eigen;


namespace MCTS
{
	struct Config
	{
		static inline float cpuct = 2;
		static inline float dirichlet_alpha = 1;
		static inline float n_simulations = 800;
		static inline float temp = 1;

		//for parallel simulate
		static inline bool parallel = true;
		static inline float vloss = 1;

		Config(cxxopts::ParseResult result){
			this->cpuct = result["cpuct"].as<float>();
			this->dirichlet_alpha = result["dirichlet_alpha"].as<float>();
			this->n_simulations = result["n_simulations"].as<float>();
			this->temp = result["temp"].as<float>();
			this->parallel = result["parallel"].as<bool>();
			this->vloss = result["vloss"].as<float>();
		}

		static void addCommandLineOptions(cxxopts::Options&  options){
			options.add_options()
				("cpuct", "Cpuct",  cxxopts::value<float>()->default_value(std::to_string(cpuct)))
				("dirichlet_alpha", "Dirichilet alpha",  cxxopts::value<float>()->default_value(std::to_string(dirichlet_alpha)))
				("n_simulations", "Number of simulations",  cxxopts::value<float>()->default_value(std::to_string(n_simulations)))
				("temp", "Temperature",  cxxopts::value<float>()->default_value(std::to_string(temp)))
				("parallel", "Parallel mcts",  cxxopts::value<bool>()->default_value(std::to_string(parallel)))
				("vloss", "Virtual loss",  cxxopts::value<float>()->default_value(std::to_string(vloss)))
			;
		}
	};

	void tree_to_dot_aux(std::shared_ptr<GameState> root, std::stringstream& dot);
	void tree_to_dot(std::shared_ptr<GameState> root);
	ArrayXf simulate(std::shared_ptr<Game> game, NNWrapper& model, MCTS::Config cfg);
	ArrayXf simulate(std::shared_ptr<GameState> root, NNWrapper& model, MCTS::Config cfg);
	ArrayXf do_parallel_simulate(std::shared_ptr<GameState> root, NNWrapper& model, MCTS::Config cfg);
	ArrayXf do_simulate(std::shared_ptr<GameState> root, NNWrapper& model, MCTS::Config cfg);
	ArrayXf simulate_random(std::shared_ptr<Game> game, MCTS::Config cfg);
	ArrayXf simulate_random(std::shared_ptr<GameState> root, MCTS::Config cfg);

}

#endif 
