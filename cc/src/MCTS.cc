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


class GameState : public std::enable_shared_from_this<GameState>
{
	private:
		std::shared_ptr<Game> game;
		int action;
		bool isExpanded;

		std::map<int, std::shared_ptr<GameState>> children;

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
				std::cout << "puct action" << action<< std::endl;
				*/
				current = current->play(action);
			}

			return current;
		}

		int getBestAction(ArrayXf puct, std::shared_ptr<GameState> current){
			ArrayXf poss = current->game->getPossibleActions();
			float max = std::numeric_limits<float>::lowest();
			int action;
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
			if (this->children.find(action) == this->children.end()){
				std::unique_ptr<Game> t_unique = this->game->copy();
				std::shared_ptr<Game> t = std::move(t_unique);
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
			}
		}

		ArrayXf getchildW(){
			return this->childW;
		}

		void updateW(float v){
			this->parent->childW[this->action] -= v;
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

				std::cout<<"end for"<<std::endl;
				std::cout<<res<<std::endl;
			
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
				std::cout<< "beepis" <<std::endl;
				ArrayXf alpha = ArrayXf::Ones(p.size())*dirichlet_alpha;
				std::cout<< "alpha" << alpha << std::endl;
				std::cout<< "d_alpha" << dirichlet_alpha << std::endl;
				ArrayXf d = dirichlet_distribution(alpha);
				std::cout<< d <<std::endl;
				std::cout<< 0.75*p <<std::endl;
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
			return count/count.sum();
		}

		bool endGame(){
			return game->ended();
		}

		bool getWinner(){
			return game->getWinner();
		}

		MatrixXf canonicalBoard(){
			return this->game->getBoard();
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
			std::cout<< "DDDDD" << dirichlet_alpha << std::endl;

		}

		ArrayXf simulate(std::shared_ptr<GameState> root, NNWrapper& model, float temp = 1, int n_simulations = 5){
			std::shared_ptr<GameState> leaf; 

			for (int i = 0; i < n_simulations; i++){
				//std::cout << "Begining sim: " << i << std::endl;
				leaf = root->select(this->cpuct);

				if (leaf->endGame()){
					leaf->backup(leaf->getWinner());
					continue;
				}

				NN::Output res = model.predict({leaf->canonicalBoard()});
				leaf->expand(res.policy, dirichlet_alpha);
				leaf->backup(res.value);
				//std::cout << "end sim: " << i << std::endl;
			}

			return root->getSearchPolicy(temp);

		}
		/*
		ArrayXf threaded_simulate(std::shared_ptr<GameState> root, NNWrapper& model, float temp = 1, int n_simulations = 5){
			int n_threads = 3;
			int simulations_to_run = n_simulations / n_threads; 
			std::vector<GameState> leafs;
			//we do more than the n_simulations currently
			for(int i = 0; i < simulations_to_run + 1; i++){

				if (leafs.size == 0){
					break
				}
				std::vector<NN::Output> = model.predict(leafs); 

			}

			return root->getSearchPolicy(temp);
		}*/



};
/*
int main(){
	std::shared_ptr<TicTacToe> t = std::make_shared<TicTacToe>(3,1);
	
	NNWrapper model = NNWrapper("../traced_model.pt");
	
	MCTS m = MCTS(3, 0.3);

	int action;

	
	std::shared_ptr<GameState> fakeparentparent;
	std::shared_ptr<GameState> fakeparent = std::make_shared<GameState>(t, 0, fakeparentparent);
	while (not t->ended()){
		std::shared_ptr<GameState> root = std::make_shared<GameState>(t, 0, fakeparent);
		std::cout << root->childN << std::endl;
		std::cout << root->childP << std::endl;
		std::cout << root->childW << std::endl;
		ArrayXf p = m.simulate(root, model, 1);
		p.maxCoeff(&action);
	//	root = root->getChildGameState(action);
		std::cout << "action probs" << p  << std::endl; 
		std::cout << "mtcs picked " << action  << std::endl; 
		t->play(action);
		t->printBoard();
		std::cout << "pick an action "; 
		std::cin >>  action;
		t->play(action);
		t->printBoard();
	//	root = root->getChildGameState(action);
	}
};

*/
#endif 