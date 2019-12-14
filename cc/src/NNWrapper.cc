#include <iostream>
#include "utils.h"
#include "NNWrapper.h"
#include "eigen/Eigen/Dense"
#include "eigen/Eigen/Core"
#include <torch/script.h>
       
using namespace Eigen;

NNWrapper::NNWrapper(std::string filename){
	try {
		this->module = torch::jit::load(filename);
	}
	catch (const c10::Error& e) {
		std::cerr << "error loading the model\n";
	}
}

void NNWrapper::reload(std::string filename){
	torch::jit::script::Module previous_module = this->module;
	try {
		this->module = torch::jit::load(filename);
	}
	catch (const c10::Error& e) {
		std::cerr << "error reloading the model, using old model\n";
		this->module = previous_module;
	}
}


NN::Output NNWrapper::predict(std::vector<MatrixXf> boards){
	//Convert boards to tensors
	std::vector<torch::Tensor> inputs_vec;
	for (auto &board : boards){
		auto torch_board = utils::eigen2libtorch(board);
		inputs_vec.push_back(torch_board);
	}
	at::Tensor input_ = torch::cat(inputs_vec);

	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(input_ );
	
	auto output = this->module.forward(inputs).toTuple()->elements();
	
	float value = output[0].toTensor().data_ptr<float>()[0];
	
	auto p = output[1].toTensor();
	Eigen::Map<ArrayXf> policy(p.data_ptr<float>(), p.size(1));
	
	NN::Output o = {
	.value = value,
	.policy = policy
	};
	
	return o;
}
		

