#include "Player.h"
#include "../eigen/Eigen/Dense"
#include "../eigen/Eigen/Core"
#include <iostream>
#include <random>
#include "NNWrapper.h"

std::unique_ptr<Player> Player::create(std::string type){
 /*	if (type == "TICTACTOE") 
        return std::make_unique<HumanPlayer>(); 
    else if (type == "CONNECTFOUR") 
        return std::make_unique<AlphaZeroPlayer>(); */
 }

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
	
	ArrayXf p = this->mcts.simulate(root, this->nn, 1, 25);
	p.maxCoeff(&action);
	
	return action; 
}
/*
NNPlayer::NNPlayer(NNWrapper nn): nn(nn){}

int NNPlayer::getAction(std::shared_ptr<Game> game){
	int action = 1;
	NN::Output res = model.predict({leaf->canonicalBoard()});

	return action; 
}
*/
int RandomPlayer::getAction(std::shared_ptr<Game> game){
	std::random_device rd;
    std::mt19937 gen(rd());

    ArrayXf poss = game->getPossibleActions();
    poss = poss/poss.sum();
    std::discrete_distribution<> dist(poss.data(),poss.data() +  poss.size());

    return dist(gen);
} 
