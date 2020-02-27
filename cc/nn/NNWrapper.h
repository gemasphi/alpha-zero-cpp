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

struct GlobalBatch
{
	std::vector<std::shared_ptr<GameState>> batch; 
	std::unordered_map<unsigned int, std::promise<std::vector<NN::Output>>> sections;

	void insertBatch(std::vector<std::shared_ptr<GameState>> leafs, std::promise<std::vector<NN::Output>> res_prom){
		this->sections.insert({this->batch.size(), std::move(res_prom)});
		this->batch.insert(this->batch.end(), leafs.begin(), leafs.end());
		
	}

	bool isFull(int size){
		return this->batch.size() >= size;
	}

	void returnValuesBySection(std::vector<NN::Output>& res, unsigned int size){	
		for (auto& prom: this->sections) {
			std::vector<NN::Output>::const_iterator first = res.begin() + prom.first;
			std::vector<NN::Output>::const_iterator last = res.begin() + prom.first + size;
			std::vector<NN::Output> res_slice(first, last);
			prom.second.set_value(res_slice);
		}

		this->sections.clear();
		this->batch.clear();
	}
};

class NNWrapper{
	private:
		GlobalBatch buffer; 

		torch::jit::script::Module module;
		torch::Device device;
		mutable std::shared_mutex modelMutex;
		
		std::string filename;
		fs::file_time_type modelLastUpdate;
		//std::unique_ptr<NNObserver> observer; 
		
		std::unordered_map<std::string, NN::Output> netCache;
		
		bool inCache(std::string board);
		NN::Output getCachedEl(std::string board);
		void inserInCache(std::string board, NN::Output o);
		void load(std::string filename);

	public:
		NNWrapper(std::string filename);
		NN::Input prepareInput(std::vector<std::shared_ptr<GameState>> leafs);

		NN::Output maybeEvaluate(std::shared_ptr<GameState> leaf);
		std::future<std::vector<NN::Output>> maybeEvaluate(std::vector<std::shared_ptr<GameState>> leafs, int globalBatching);
		std::vector<NN::Output> predict(NN::Input input);

		//std::shared_mutex* getModelMutex();
		void shouldLoad(std::string filename);
		std::string getFilename();

		
};

#endif 