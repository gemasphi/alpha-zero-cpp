#ifndef NNWRAPPER_H     
#define  NNWRAPPER_H

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <torch/script.h>
#include "NNUtils.h"
#include <GameState.h>

#include <sys/inotify.h>
#include <fcntl.h>

#include <thread>
//#include <atomic>

using namespace Eigen;

class NNWrapper{
	private:
		torch::jit::script::Module module;
		std::unordered_map<std::string, NN::Output> netCache;
		
		std::thread fileWatcher;
		omp_lock_t modelock; //TODO: not freed rn

		bool inCache(std::string board);
		NN::Output getCachedEl(std::string board);
		void inserInCache(std::string board, NN::Output o);
		void setup_inotify(std::string directory);

	public:
		NNWrapper(std::string filename);
		std::vector<NN::Output> predict(NN::Input input);
		NN::Output maybeEvaluate(std::shared_ptr<GameState> leaf);
		std::vector<NN::Output> maybeEvaluate(std::vector<std::shared_ptr<GameState>> leafs);
		void load(std::string filename);

		
};

#endif 