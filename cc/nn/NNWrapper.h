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
	struct Section{
		unsigned int first;
		unsigned int last;
		std::shared_ptr<std::promise<std::vector<NN::Output>>> promise; 

		Section(unsigned int first, unsigned int last, std::shared_ptr<std::promise<std::vector<NN::Output>>> promise):
				first(first), last(last), promise(promise){}

	};

	std::vector<std::shared_ptr<GameState>> batch; 
	std::vector<Section> sections;
	int maxAdds;
	int size;

	GlobalBatch(int maxAdds, int size) : maxAdds(maxAdds), size(size){}

	void insertBatch(std::vector<std::shared_ptr<GameState>> leafs, std::shared_ptr<std::promise<std::vector<NN::Output>>> res_prom){
		Section sec(this->batch.size(), this->batch.size() + leafs.size(), res_prom); 

		this->sections.push_back(sec);
		this->batch.insert(this->batch.end(), leafs.begin(), leafs.end());
	}

	bool isFull(){
		return this->batch.size() >= size || this->sections.size() == maxAdds;
	}

	void returnValuesBySection(std::vector<NN::Output>& res){	
		for(auto& sec : this->sections ){
			auto first = res.begin() + sec.first;
			auto last = res.begin() + sec.last;
			std::vector<NN::Output> res_slice(first, last);
			sec.promise->set_value(res_slice);
		}

		this->sections.clear();
		this->batch.clear();
	}
};

class NNWrapper{
	private:
		GlobalBatch& buffer; 

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
		NNWrapper(std::string filename, GlobalBatch& buffer);
		NN::Input prepareInput(std::vector<std::shared_ptr<GameState>> leafs);

		NN::Output maybeEvaluate(std::shared_ptr<GameState> leaf);
		std::future<std::vector<NN::Output>> maybeEvaluate(std::vector<std::shared_ptr<GameState>> leafs, int globalBatching);
		std::vector<NN::Output> predict(NN::Input input);

		void flushBuffer();
		
		//std::shared_mutex* getModelMutex();
		void shouldLoad(std::string filename);
		std::string getFilename();

		
};

#endif 