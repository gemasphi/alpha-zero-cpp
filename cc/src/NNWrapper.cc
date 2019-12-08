#include <iostream>
#include "utils.h"
#include "NNWrapper.h"
#include "games/eigen/Eigen/Dense"
#include "games/eigen/Eigen/Core"
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


NN::Output NNWrapper::predict(MatrixXf board){
	auto torch_board = utils::eigen2libtorch(board);

	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(torch_board);
	
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
		

