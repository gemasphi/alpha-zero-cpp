#ifndef MCTS_H_
#define MCTS_H_

#include <iostream>
#include <map>
#include <NNWrapper.h>
#include <Game.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <memory>
#include <limits> 
#include "GameState.h"

using namespace Eigen;


class MCTS
{
	private:
		float cpuct;
		float dirichlet_alpha;

	public:
		MCTS(float cpuct, float dirichlet_alpha);
		ArrayXf simulate(std::shared_ptr<Game> game, NNWrapper& model, float temp = 1, int n_simulations = 25);
		//TODO: THis should be inside NNWrapper 

		/*
		ArrayXf threaded_simulate(std::shared_ptr<GameState> root, NNWrapper& model, float temp = 1, int n_simulations = 5){
			int eval_size = 4;
			int simulations_to_run = n_simulations / eval_size; 
			std::vector<GameState> leafs;
			//we do more than the n_simulations currently
			
			for(int i = 0; i < 5 + 1; i++){
				#pragma omp parallel
				{
					GameState leaf = root->select(this->cpuct);

					if (leaf->endGame()){
						leaf->backup(leaf->getWinner());
						continue;
					}

					leaf->addVirtualLoss();
					leafs.push_back(leaf);
				}

				if (leafs.size == 0){
					break;
				}
				
				NN::Output output = model.predict(leafs); 
   				#pragma omp for
   				{
   					for (NN::Output o& : output){
						leaf.remove_virtual_loss();
						leaf.expand(o.p, dirichlet_alpha);
						leaf.backup(o.v);
   					}
				}
			}
			return root->getSearchPolicy(temp);
		}
			*/



};

#endif 
