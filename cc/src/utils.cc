#include "eigen/Eigen/Dense"
#include "eigen/Eigen/Core"
#include <torch/script.h>
#include <random>
#include "utils.h"

using namespace Eigen;

template<typename V>
	Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic> utils::libtorch2eigen(torch::Tensor &Tin) {
		/*
		 LibTorch is Row-major order and Eigen is Column-major order.
		 MatrixXrm uses Eigen::RowMajor for compatibility.
		 */
		auto T = Tin.to(torch::kCPU);
		Eigen::Map<utils::MatrixXrm<V>> E(T.data_ptr<V>(), T.size(0), T.size(1));
		return E;
	};

template <typename V>
	  torch::Tensor utils::eigen2libtorch(utils::MatrixX<V> &M) {
	    Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> E(M);
	    std::vector<int64_t> dims = {E.rows(), E.cols()};
	    auto T = torch::from_blob(E.data(), dims).clone(); //.to(torch::kCPU);
	    return T;
	};	
	
ArrayXf utils::dirichlet_distribution(ArrayXf& alpha){
	std::random_device rd;
	std::mt19937 gen(rd());
	ArrayXf res(alpha.size());

	for(int i; i < alpha.size(); i++){
		std::gamma_distribution<double> dist(alpha(i),1);
		auto sample = dist(gen);
		res(i) = sample;
	}

	return res/res.sum();
} 

