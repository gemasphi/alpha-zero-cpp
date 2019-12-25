#include <iostream>
#include "NNWrapper.h"
#include "eigen/Eigen/Dense"
#include "eigen/Eigen/Core"
#include <torch/script.h>
       
using namespace Eigen;

NNWrapper::NNWrapper(std::string filename){
	try {
		this->module = torch::jit::load(filename, torch::kCUDA);
	}
	catch (const c10::Error& e) {
		std::cerr << "error loading the model\n";
	}
}

void NNWrapper::reload(std::string filename){
	torch::jit::script::Module previous_module = this->module;
	try {
		this->module = torch::jit::load(filename, torch::kCUDA);
	}
	catch (const c10::Error& e) {
		std::cerr << "error reloading the model, using old model\n";
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
		

