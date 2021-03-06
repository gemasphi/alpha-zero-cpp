#include "NNWrapper.h"

using namespace Eigen;

NNWrapper::NNWrapper(std::string filename, std::unique_ptr<GlobalBatch> buffer) : buffer(std::move(buffer)), device(torch::kCPU){
	if (torch::cuda::is_available()) {
  		std::cout << "CUDA is available! Training on GPU." << std::endl;
  		this->device = torch::kCUDA;
	}

	//this->observer = 
	//	 std::make_unique<NNObserver>(*this, filename, watchFile);;
	this->load(filename);
}

NNWrapper::NNWrapper(std::string filename): NNWrapper(filename, {}){}

void NNWrapper::decreaseBufferSize(){
	this->buffer->maxAdds--;
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

NN::Input NNWrapper::prepareInput(std::vector<std::shared_ptr<GameState>> leafs){
	std::vector<std::vector<MatrixXf>> _boards;

	for (auto leaf: leafs){
		_boards.push_back(leaf->getNetworkInput());
	}	

	return NN::Input(_boards);
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



std::future<std::vector<NN::Output>> NNWrapper::maybeEvaluate(std::vector<std::shared_ptr<GameState>> leafs,  
								int globalBatchSize){
	
	auto res_prom = 
		std::make_shared<std::promise<std::vector<NN::Output>>>();

	auto res_future = res_prom->get_future();

    if (this->buffer){
		#pragma omp critical(batch)
		{
			this->buffer->insertBatch(leafs, res_prom);	

			if (this->buffer->isFull()){
				this->flushBuffer();
			}
		}

    }
    else{
		std::vector<NN::Output> res = this->predict(this->prepareInput(leafs));
		res_prom->set_value(res);
    }

    return res_future;	
}

void NNWrapper::flushBuffer(){
	if (this->buffer->batch.size() != 0){
		std::vector<NN::Output> res = this->predict(this->prepareInput(this->buffer->batch));
		this->buffer->returnValuesBySection(res);
	}
}