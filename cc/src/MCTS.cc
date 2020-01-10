#ifndef MCTS_CC_
#define MCTS_CC_

#include <iostream>
#include <map>
#include "NNWrapper.h"
#include <Game.h>
#include "eigen/Eigen/Dense"
#include "eigen/Eigen/Core"
#include "utils.h"
#include <memory>
#include <torch/script.h>
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

		GameState(std::shared_ptr<Game> game, int action, std::shared_ptr<GameState> parent){
			this->isExpanded = false;;
			this->parent = parent;
			this->game = game;
			this->action = action;
			this->childW = ArrayXf::Zero(game->getActionSize());
			this->childP = ArrayXf::Zero(game->getActionSize());
			this->childN = ArrayXf::Zero(game->getActionSize());
			this->children = std::vector<std::shared_ptr<GameState>>(game->getActionSize());  
		}

		void addVirtualLoss(){
			std::shared_ptr<GameState> current = shared_from_this();

			while (current->getParent()){
				current->updateW(1);
				current = current->getParent();
			}
		}

		void removeVirtualLoss(){
			std::shared_ptr<GameState> current = shared_from_this();

			while (current->getParent()){
				current->updateW(-1);
				current = current->getParent();
			}
		}

		std::shared_ptr<GameState> getParent(){
			return this->parent;
		} 

		std::shared_ptr<GameState> select(int cpuct){
			std::shared_ptr<GameState> current = shared_from_this();
			int action;
			ArrayXf puct;//, q, u;
			
			while (current->isExpanded and not current->game->ended()){
				/*q = current->childQ();
				u = current->childU(cpuct);
				*/
				puct = current->childQ() + current->childU(cpuct);
				action = getBestAction(puct, current);
				
				/*std::cout << "puct" << puct<< std::endl;
				std::cout << "puct q" << q<< std::endl;
				std::cout << "puct u" << u<< std::endl;
				std::cout << "puct action" << action<< std::endl;*/
				
				current = current->play(action);
			}

			return current;
		}

		int getBestAction(ArrayXf puct, std::shared_ptr<GameState> current){
			ArrayXf poss = current->game->getPossibleActions();
			float max = std::numeric_limits<float>::lowest();
			int action = -1;
			for (int i; i < poss.size(); i++){
				if (poss[i] != 0){
					if(puct[i] > max){
						action = i; 	
						max = puct[i];	
					}	
				}				
			}
						
			return action;			
		}

		std::shared_ptr<GameState> play(int action){
			if (!this->children[action]){
				std::shared_ptr<Game> t = std::move(this->game->copy());
				t->play(action);

				this->children[action] = std::make_shared<GameState>(t, action, shared_from_this()); 
			}

			return this->children[action];
		}

		void backup(float v){
			std::shared_ptr<GameState> current = shared_from_this();
			while (current->getParent()){
				current->incN();
				current->updateW(v);
				current = current->getParent();
				v *= -1;
			}
		}

		ArrayXf getchildW(){
			return this->childW;
		}

		void updateW(float v){
			this->parent->childW[this->action] += v;
		}

		void incN(){
			this->parent->childN[this->action]++;
		}
	
		float getN(){
			return this->parent->childN(this->action);
		}

		float getP(){
			return this->parent->childP(this->action);
		}
		
		ArrayXf childQ(){
			return this->childW / (ArrayXf::Ones(game->getActionSize()) + this->childN);
		}

		ArrayXf childU(float cpuct){
			return cpuct 
					* this->childP 
					* ( 
						std::sqrt(this->getN())
						/
						(ArrayXf::Ones(game->getActionSize()) + this->childN)
					   );
		}

		ArrayXf dirichlet_distribution(ArrayXf alpha){
			ArrayXf res =  ArrayXf::Zero(alpha.size());
			std::random_device rd;
			std::mt19937 gen(rd());

			for(int i; i < alpha.size(); i++){
				std::gamma_distribution<double> dist(alpha(i),1);
				auto sample = dist(gen); 
				res(i) = sample;
			}
			
			if (res.sum() == 0){
				return res; //TODO: hacky 
			} else {
				return res/res.sum();
			}	
		} 


		void expand(ArrayXf p, float dirichlet_alpha){
			this->isExpanded = true;
			
			ArrayXf poss = this->game->getPossibleActions();

			if (!this->parent->parent){
				ArrayXf alpha = ArrayXf::Ones(p.size())*dirichlet_alpha;
				ArrayXf d = dirichlet_distribution(alpha);
				p = 0.75*p + 0.25*d;
			}

			this->childP = this->getValidActions(p, poss);

			//std::cout << "p inside " << p << std::endl;
			//std::cout << "poss " << poss << std::endl;
			//std::cout << "childp" << this->childP << std::endl;
		}

		ArrayXf getValidActions(ArrayXf pred, ArrayXf poss){
			ArrayXf valid_actions = (pred * poss);

			if (not valid_actions.any()){
				valid_actions = poss;
			}

			return valid_actions/valid_actions.sum();
		}

		ArrayXf getSearchPolicy(float temp){
			/*std::cout << "W" << std::endl;
			std::cout << childW << std::endl;
			std::cout << "N" << std::endl;
			std::cout << childN << std::endl;
			std::cout << "P" << std::endl;
			std::cout << childP << std::endl;*/
			ArrayXf count = pow(this->childN,1/temp);
			//std::cout << "childN " << this->childN << std::endl;
			//std::cout << "count " << count << std::endl;
			//std::cout << "count.sum " << count.sum()<< std::endl;
			return count/count.sum();
		}

		bool endGame(){
			return game->ended();
		}

		bool getPlayer(){
			return game->getPlayer();
		}
		bool getWinner(){
			return game->getWinner();
		}

		MatrixXf getCanonicalBoard(){
			return this->game->getBoard()*this->game->getPlayer();
		}

		//Todo: this shouldnt belong to this class
		NN::Input getNetworkInput(){
			std::vector<MatrixXf> game_state ;
			auto dims = this->game->getBoardSize();
			std::shared_ptr<GameState> current = shared_from_this();
			game_state.insert(game_state.begin(), current->getCanonicalBoard());

			for(int i = 0; i < this->game->getInputPlanes() - 1; i++){
				if (current->getParent()){
					current =  current->getParent();
					game_state.insert(game_state.begin(), current->getCanonicalBoard());
				}
				else{
					game_state.insert(game_state.begin(), MatrixXf::Zero(dims[0], dims[1]));
				}
			
			}
			return NN::Input(game_state);
		}

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
		MCTS(float cpuct, float dirichlet_alpha){
			this->cpuct = cpuct;
			this->dirichlet_alpha = dirichlet_alpha;
		}

		ArrayXf simulate(std::shared_ptr<GameState> root, NNWrapper& model, float temp = 1, int n_simulations = 25){
			std::shared_ptr<GameState> leaf; 

			for (int i = 0; i < n_simulations; i++){
				leaf = root->select(this->cpuct);
				

				if (leaf->endGame()){
					leaf->backup(leaf->getWinner()*root->getPlayer());
					continue;
				}

				NN::Output res = maybeEvaluate(model, leaf);
				
				leaf->expand(res.policy, dirichlet_alpha);
				leaf->backup(res.value);
			}
			return root->getSearchPolicy(temp);

		}

		//TODO: THis should be inside NNWrapper 
		NN::Output maybeEvaluate(NNWrapper& model, std::shared_ptr<GameState> leaf){
			std::stringstream ss;
    		ss << leaf->getCanonicalBoard();
    		std::string board = ss.str();	
    		

			if (model.inCache(board)) {
				return model.getCachedEl(board);
			}
			else{
				NN::Output res = model.predict(leaf->getNetworkInput())[0];
				
				#pragma omp critical(netCache)
				model.inserInCache(board, res);
				
				return res;
			}

		}

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