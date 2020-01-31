#ifndef NNWRAPPER_H     
#define  NNWRAPPER_H

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <torch/script.h>
#include "NNUtils.h"
#include <GameState.h>

using namespace Eigen;

class NNWrapper{
	private:
		torch::jit::script::Module module;
		std::unordered_map<std::string, NN::Output> netCache;
	
		bool inCache(std::string board);
		NN::Output getCachedEl(std::string board);
		void inserInCache(std::string board, NN::Output o);
	
	public:
		NNWrapper(std::string filename);
		std::vector<NN::Output> predict(NN::Input input);
		NN::Output maybeEvaluate(std::shared_ptr<GameState> leaf);
		std::vector<NN::Output> maybeEvaluate(std::vector<std::shared_ptr<GameState>> leafs);
		void reload(std::string filename);

		
};

#endif 