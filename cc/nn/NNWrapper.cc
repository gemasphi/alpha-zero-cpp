#include "NNWrapper.h"

using namespace Eigen;
NNWrapper::NNWrapper(std::string filename){
	try {
		std::cout << "loading the model\n";
		this->module = torch::jit::load(filename, torch::kCUDA);
	}
	catch (const c10::Error& e) {
		std::cout << "error loading the model\n";
	}
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

void NNWrapper::reload(std::string filename){
	torch::jit::script::Module previous_module = this->module;
	try {
		std::cout << "reloading the model\n";
		this->module = torch::jit::load(filename, torch::kCUDA);
		netCache = std::unordered_map<std::string, NN::Output>();
	}
	catch (const c10::Error& e) {
		std::cout << "error reloading the model, using old model\n";
		this->module = previous_module;
	}
}

std::vector<NN::Output> NNWrapper::predict(NN::Input input){
	std::vector<torch::jit::IValue> jit_inputs;
	jit_inputs.push_back(input.boards.to(at::kCUDA));
	torch::jit::IValue output = this->module.forward(jit_inputs);
	std::vector<NN::Output> o;

	for(int i = 0; i < input.batch_size; i++){
		o.push_back(NN::Output(output, i));
	} 

	return o;
}
