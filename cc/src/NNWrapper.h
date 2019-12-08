#include <iostream>
#include "games/eigen/Eigen/Dense"
#include "games/eigen/Eigen/Core"
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
		NN::Output predict(MatrixXf board);
};

