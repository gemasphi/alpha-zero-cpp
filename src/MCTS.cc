#include <iostream>
#include <map>
#include "games/Tictactoe.h"
#include "games/eigen/Eigen/Dense"
#include "games/eigen/Eigen/Core"
#include <memory>
using namespace Eigen;

class GameState
{
	private:
		GameState* parent;
		TicTacToe* game;
		int action;
		bool isExpanded;

		std::map<int, GameState*> children;
		ArrayXf childW;
		ArrayXf childP;
		ArrayXf childN;

	public:
		GameState(TicTacToe* game, int action, GameState* parent){
			this->isExpanded = false;;
			this->parent = parent;
			this->game = game;
			this->action = action;
			this->childW = ArrayXf::Zero(game->getActionSize());
			this->childP = ArrayXf::Zero(game->getActionSize());
			this->childN = ArrayXf::Zero(game->getActionSize());
		}

		void addVirtualLoss(){
			GameState* current(this);

			while (current->getParent() != nullptr){
				current->updateW(1);
				current = current->getParent();
			}
		}

		void removeVirtualLoss(){
			GameState* current(this);

			while (current->getParent() != nullptr){
				current->updateW(-1);
				current = current->getParent();
			}
		}

		GameState* getParent(){
			return this->parent;
		} 

		GameState* select(int cpuct){
			GameState* current(this);
			int action;
			ArrayXf puct;
			
			while (current->isExpanded and not current->game->ended()){
				puct = current->childQ() + current->childU(cpuct); 
				puct.maxCoeff(&action);

				current = current->play(action);
			}

			return current;
		}

		GameState* play(int action){
			if (this->children.find(action) == this->children.end()){
				TicTacToe* t(new TicTacToe(*game));
				GameState* parent(this); 
				t->play(action);
				this->children[action] = new GameState(t, action, parent); 
			}

			return this->children[action];
		}

		void backup(float v){
			GameState* current(this);
			while (current->getParent() != nullptr){
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
			if (parent == nullptr){
				return 0;
			}
			else{
				return this->parent->childN(this->action);
			}
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

		void expand(ArrayXf p){
			this->isExpanded = true;
			ArrayXf poss = this->game->getPossibleActions();
			this->childP = this->getValidActions(p, poss);
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

		MatrixXi canonicalBoard(){
			return this->game->getBoard();
		}
};


namespace NN{
	struct Output {
		float value;
		ArrayXf policy;
	};

	Output predict(MatrixXi board){
		Output o = {
			.value = 0.3,
			.policy = ArrayXf::Random(9)
		};

		return o;
	}
}

class MCTS
{
	private:
		float cpuct;
		float dirichlet_alpha;
		int n_simulations;

	public:
		MCTS(int cpuct, int dirichlet_alpha, int n_simulations){
			this->cpuct = cpuct;
			this->dirichlet_alpha = dirichlet_alpha;
			this->n_simulations = n_simulations;
		}

		ArrayXf simulate(TicTacToe* game, float temp){
			GameState* root(new GameState(game, 0, (GameState*) nullptr));
			GameState* leaf; 
			 
			for (int i = 0; i < n_simulations; i++){
				//std::cout << "Begining sim: " << i << std::endl;
				leaf = root->select(this->cpuct);

				if (leaf->endGame()){
					leaf->backup(leaf->getWinner());
					continue;
				}
				NN::Output res = NN::predict(leaf->canonicalBoard());
				leaf->expand(res.policy);
				leaf->backup(res.value);
				//std::cout << "end sim: " << i << std::endl;
			}

			return root->getSearchPolicy(temp);

		}
};

int main(){
	MCTS m = MCTS(1, 0.3, 15);
	TicTacToe* t(new TicTacToe(3,1));
	int action;
	while (not t->ended()){
		ArrayXf p = m.simulate(t, 1);
		p.maxCoeff(&action);
		std::cout << "action probs" << p  << std::endl; 
		std::cout << "mtcs picked " << action  << std::endl; 
		t->play(action);
		t->printBoard();
		std::cout << "pick an action "; 
		std::cin >>  action;
		t->play(action);
		t->printBoard();
	}
};

