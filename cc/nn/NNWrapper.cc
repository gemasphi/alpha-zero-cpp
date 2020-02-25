#include "NNWrapper.h"

using namespace Eigen;


NNWrapper::NNWrapper(std::string filename) : device(torch::kCPU){
	if (torch::cuda::is_available()) {
  		std::cout << "CUDA is available! Training on GPU." << std::endl;
  		this->device = torch::kCUDA;
	}

	//this->observer = 
	//	 std::make_unique<NNObserver>(*this, filename, watchFile);;  
	this->load(filename);
}
	
void NNWrapper::shouldLoad(std::string filename){
	fs::file_time_type last_update = fs::last_write_time(filename);

	if (this->modelLastUpdate != last_update){
		std::cout << "New model found" << std::endl;
		this->load(filename);

	}
}	

std::string NNWrapper::getFilename(){
	return this->filename;
}

/*
std::shared_mutex* NNWrapper::getModelMutex(){
	return &(this->modelMutex);
}*/

NN::Output NNWrapper::maybeEvaluate(std::shared_ptr<GameState> leaf){
	std::stringstream ss;
	ss << leaf->getCanonicalBoard();
	std::string board = ss.str();	
	

	if (this->inCache(board)) {
		return this->getCachedEl(board);
	}
	else{
		NN::Output res = this->predict(NN::Input(leaf->getNetworkInput()))[0];
		
		//pragma omp critical(netCache)
		this->inserInCache(board, res);
		
		return res;
	}

}

void NNWrapper::maybeEvaluate(std::vector<std::shared_ptr<GameState>> leafs,  
								std::promise<std::vector<NN::Output>> &result){
	
	#pragma omp critical(batch)
	{
		this->batchSection.insert({this->batch.size(), result});
		this->batch.insert(this->batch.end(), leafs.begin(), leafs.end());
	}

	#pragma omp critical(batch)
	if (this->batch.size() >= this->batchSize){
		std::vector<NN::Output> res 
			= this->maybeEvaluate(this->batch);

		for (auto& prom: this->batchSection) {
			std::vector<NN::Output>::const_iterator first = res.begin() + prom.first;
			std::vector<NN::Output>::const_iterator last = res.begin() + prom.first + leafs.size();
			std::vector<NN::Output> res_slice(first, last);
			prom.second.set_value(res_slice);
		}
		this->batchSection.clear();
		this->batch.clear();
	}
}


std::vector<NN::Output> NNWrapper::maybeEvaluate(std::vector<std::shared_ptr<GameState>> leafs){
	std::vector<std::vector<MatrixXf>> _boards;

	for (auto leaf: leafs){
		_boards.push_back(leaf->getNetworkInput());
	}	

	std::vector<NN::Output> res 
		= this->predict(NN::Input(_boards));

	return res;
}


bool NNWrapper::inCache(std::string board){
	return this->netCache.find(board) != this->netCache.end();
}

NN::Output NNWrapper::getCachedEl(std::string board){
	return this->netCache.at(board);
}

void NNWrapper::inserInCache(std::string board, NN::Output o){
	this->netCache.insert({board, o});
}

void NNWrapper::load(std::string filename){
	std::unique_lock lock(this->modelMutex);
	try {
		std::cout << "loading the model\n";
		this->module = torch::jit::load(filename, this->device);
		this->modelLastUpdate = fs::last_write_time(filename);
		this->filename = filename;
		netCache = std::unordered_map<std::string, NN::Output>();
		std::cout << "model loaded\n";
	}
	catch (const c10::Error& e) {
		std::cout << "error reloading the model, using old model\n";
	}

	lock.unlock();
}

std::vector<NN::Output> NNWrapper::predict(NN::Input input){
	std::vector<torch::jit::IValue> jit_inputs;
	jit_inputs.push_back(input.boards.to(this->device));
	
	std::shared_lock lock(this->modelMutex);
	torch::jit::IValue output = this->module.forward(jit_inputs);
	lock.unlock();

	std::vector<NN::Output> o;

	for(int i = 0; i < input.batch_size; i++){
		o.push_back(NN::Output(output, i));
	} 

	return o;
}
