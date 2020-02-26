#ifndef NNWRAPPER_H     
#define  NNWRAPPER_H

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <torch/script.h>
#include <torch/torch.h>
#include "NNUtils.h"
#include <GameState.h>
//#include "NNObserver.h"

#include <experimental/filesystem>
#include <mutex>  
#include <shared_mutex>
#include <future>

using namespace Eigen;
namespace fs = std::experimental::filesystem;

//class NNObserver;   //forward declaration

class NNWrapper{
	private:
		std::vector<std::shared_ptr<GameState>> batch; 
		unsigned int batchSize = 8; 
		std::unordered_map<unsigned int, std::promise<std::vector<NN::Output>>&> batchSection;

		torch::jit::script::Module module;
		torch::Device device;
		std::unordered_map<std::string, NN::Output> netCache;
		
		std::string filename;
		fs::file_time_type modelLastUpdate;
		mutable std::shared_mutex modelMutex;
		//std::unique_ptr<NNObserver> observer; 
		
		bool inCache(std::string board);
		NN::Output getCachedEl(std::string board);
		void inserInCache(std::string board, NN::Output o);
		void load(std::string filename);

	public:
		NNWrapper(std::string filename);
		NN::Output maybeEvaluate(std::shared_ptr<GameState> leaf);
		std::vector<NN::Output> maybeEvaluate(std::vector<std::shared_ptr<GameState>> leafs);
		void maybeEvaluate(std::vector<std::shared_ptr<GameState>> leafs,
							std::promise<std::vector<NN::Output>> & result);

		std::vector<NN::Output> predict(NN::Input input);

		//std::shared_mutex* getModelMutex();
		void shouldLoad(std::string filename);
		std::string getFilename();

		
};

#endif 