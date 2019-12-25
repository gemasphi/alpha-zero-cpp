<<<<<<< HEAD
#ifndef UTILS_H     
#define  UTILS_H
=======
>>>>>>> 4f6c57b7d58c4c7c93b660eccc6a58695228fb3b
#include "eigen/Eigen/Dense"
#include "eigen/Eigen/Core"
#include <torch/script.h>
#include <random>

using namespace Eigen;

namespace utils{
	template <typename V>
	  using MatrixXrm = typename Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

	template <typename V>
	  using MatrixX = typename Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic>;

	template<typename V>
	Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic> libtorch2eigen(torch::Tensor &Tin) {
		/*
		 LibTorch is Row-major order and Eigen is Column-major order.
		 MatrixXrm uses Eigen::RowMajor for compatibility.
		 */
		auto T = Tin.to(torch::kCPU);
		Eigen::Map<MatrixXrm<V>> E(T.data_ptr<V>(), T.size(0), T.size(1));
		return E;
	};

	template <typename V>
	  torch::Tensor eigen2libtorch(MatrixX<V> &M) {
	    Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> E(M);
	    std::vector<int64_t> dims = {E.rows(), E.cols()};
	    auto T = torch::from_blob(E.data(), dims).clone(); //.to(torch::kCPU);
	    return T;
	};	
};
#endif 