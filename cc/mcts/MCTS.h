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
using namespace Eigen;

//TODO add asserts bruv
class GameState : public std::enable_shared_from_this<GameState>
{
	private:
		std::shared_ptr<Game> game;
		int action;
		bool isExpanded;
		std::vector<std::shared_ptr<GameState>> children;

	public:
		std::shared_ptr<GameState> parent;
		ArrayXf childW;
		ArrayXf childP;
		ArrayXf childN;

		GameState(std::shared_ptr<Game> game, int action, std::shared_ptr<GameState> parent);
		void addVirtualLoss();
		void removeVirtualLoss();
		std::shared_ptr<GameState> getParent();
		std::shared_ptr<GameState> select(int cpuct);
		int getBestAction(ArrayXf puct, std::shared_ptr<GameState> current);
		std::shared_ptr<GameState> play(int action);
		void backup(float v);
		ArrayXf getchildW();
		void updateW(float v);
		void incN();
		float getN();
		float getP();
		ArrayXf childQ();
		ArrayXf childU(float cpuct);
		ArrayXf dirichlet_distribution(ArrayXf alpha);
		void expand(ArrayXf p, float dirichlet_alpha);
		ArrayXf getValidActions(ArrayXf pred, ArrayXf poss);
		ArrayXf getSearchPolicy(float temp);
		bool endGame();
		bool getPlayer();
		bool getWinner();
		MatrixXf getCanonicalBoard();

		//Todo: this shouldnt belong to this class
		NN::Input getNetworkInput();

/*		std::shared_ptr<GameState> getChildGameState(int action){
		//	if (this->children.find(action) == this->children.end()){
		}:*/
};

class MCTS
{
	private:
		float cpuct;
		float dirichlet_alpha;

	public:
		MCTS(float cpuct, float dirichlet_alpha);
		ArrayXf simulate(std::shared_ptr<GameState> root, NNWrapper& model, float temp = 1, int n_simulations = 25);
		//TODO: THis should be inside NNWrapper 
		NN::Output maybeEvaluate(NNWrapper& model, std::shared_ptr<GameState> leaf);

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
