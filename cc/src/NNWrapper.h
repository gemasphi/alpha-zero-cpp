#ifndef NNWRAPPER_H     
#define  NNWRAPPER_H

#include <iostream>
#include "eigen/Eigen/Dense"
#include "eigen/Eigen/Core"
#include <torch/script.h>

using namespace Eigen;

namespace NN{
	struct Output {
		float value;
		ArrayXf policy;
	};
}

class NNWrapper{
	private:
		torch::jit::script::Module module;

	public:
		NNWrapper(std::string filename);
		void reload(std::string filename);
		NN::Output predict(std::vector<MatrixXf> board);
};

#endif 