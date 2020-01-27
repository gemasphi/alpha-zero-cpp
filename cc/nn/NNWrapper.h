#ifndef NNWRAPPER_H     
#define  NNWRAPPER_H

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <torch/script.h>
#include "utils.h"

using namespace Eigen;

namespace NN{
	struct Output {
		float value;
		ArrayXf policy;

		Output(){};
		Output(const Output &o) {value = o.value; policy = o.policy; } 
		Output(torch::jit::IValue output, int batch_i){
			auto o = output.toTuple()->elements();
			value =  o[0].toTensor().to(torch::kCPU).data_ptr<float>()[batch_i];
			auto p = o[1].toTensor().to(torch::kCPU);

			policy = Eigen::Map<ArrayXf>(p[batch_i].data_ptr<float>(), p.size(1));
		}

		friend std::ostream& operator<<(std::ostream& os, const Output& o){
			os  << "value: " 
				<< o.value 
				<< std::endl 
				<< "policy: " 
				<< o.policy 
				<< std::endl;

			return os;
		}
	};

	struct Input {
		at::Tensor boards;
		int batch_size;
		//Single
		Input(std::vector<MatrixXf> _boards) : batch_size(1){
			std::vector<torch::Tensor> inputs_vec;
			for (auto &board : _boards){
				auto torch_board = utils::eigen2libtorch(board);
				inputs_vec.push_back(torch_board);
			}

			boards = torch::stack(inputs_vec);
		}

		//Batch
		Input(std::vector<std::vector<MatrixXf>> _boards) : batch_size(_boards.size())
		{
			std::vector<at::Tensor> inputs_vec;

			for (auto &input : _boards){
				std::vector<torch::Tensor> game_state_t;
				for (auto &board : input){
					auto torch_board = utils::eigen2libtorch(board);
					game_state_t.push_back(torch_board);
				}
				inputs_vec.push_back(torch::stack(game_state_t));
			}

			boards = torch::cat(inputs_vec);
		}

	};
}

class NNWrapper{
	private:
		torch::jit::script::Module module;
		std::unordered_map<std::string, NN::Output> netCache;
	
	public:
		NNWrapper(std::string filename);
		void reload(std::string filename);
		std::vector<NN::Output> predict(NN::Input input);
		bool inCache(std::string board);
		NN::Output getCachedEl(std::string board);
		void inserInCache(std::string board, NN::Output o);

};

#endif 