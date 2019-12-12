#include "Player.h"
#include "../eigen/Eigen/Dense"
#include "../eigen/Eigen/Core"
#include <iostream>

int HumanPlayer::getAction(std::shared_ptr<Game> game){
	int action;
	ArrayXf poss = game->getPossibleActions();
	std::cout << "Enter an action";  
	std::cin >> action;
	action--;

	while (poss[action] != 1){
		std::cout << "Invalid, Enter an action";  
		std::cin >> action;
		action--;
	}  

	return action;
}

AlphaZeroPlayer::AlphaZeroPlayer(NNWrapper nn, MCTS mcts): nn(nn), mcts(mcts){}

int AlphaZeroPlayer::getAction(std::shared_ptr<Game> game){
	int action;
	std::shared_ptr<GameState> fakeparentparent;
	std::shared_ptr<GameState> fakeparent = std::make_shared<GameState>(game, 0, fakeparentparent);
	std::shared_ptr<GameState> root = std::make_shared<GameState>(game, 0, fakeparent);
	
	ArrayXf p = this->mcts.simulate(root, this->nn);
	p.maxCoeff(&action);
	
	return action;
}