#include "Player.h"
#include "../eigen/Eigen/Dense"
#include "../eigen/Eigen/Core"
#include <iostream>
#include <random>
#include "NNWrapper.h"
#include <string>

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

ConnectSolver::ConnectSolver(std::string opening_book){
  this->solver.loadBook(opening_book);	
}

/*
pos = (encontrar o menor valor) * -1/ encontrar maior
neg = encontrar o maior valor
mistura = encontrar maior valor 

*/
int ConnectSolver::getAction(std::shared_ptr<Game> game){
	std::shared_ptr<ConnectFour> c_game = std::dynamic_pointer_cast<ConnectFour>(game); 

    ArrayXf poss = c_game->getPossibleActions();
    int count_poss = (poss != 0).count();
    ArrayXf actions = ArrayXf::Zero(count_poss);
    ArrayXf scores = ArrayXf::Zero(count_poss);
    int added = 0; 
	int score;
	for (int i = 0; i < poss.size(); i++){
		if (poss[i] != 0){
			Position pos;
			std::string to_play = c_game->getPlayedMoves()+ std::to_string(i + 1);
    		
    		int moves_played = pos.play(to_play);
    		if (moves_played != to_play.size()){
    			return i;
    		}

    		score = this->solver.solve(pos, false)*-1;

			actions[added] = i;
			scores[added] = score;
    		added ++;
    		
    	}
    }

    int max_score;
    scores.maxCoeff(&max_score);

    return actions[max_score];
}

/*

NNPlayer::NNPlayer(NNWrapper nn): nn(nn){}

int AlphaZeroPlayer::getAction(std::shared_ptr<Game> game){
	int action;
	std::shared_ptr<GameState> fakeparentparent;
	std::shared_ptr<GameState> fakeparent = std::make_shared<GameState>(game, 0, fakeparentparent);
	std::shared_ptr<GameState> root = std::make_shared<GameState>(game, 0, fakeparent);
	
	ArrayXf p = this->mcts.simulate(root, this->nn, 1, 125);
	p.maxCoeff(&action);
	
	return action; 
}1
*/
AlphaZeroPlayer::AlphaZeroPlayer(NNWrapper nn, MCTS mcts): nn(nn), mcts(mcts){}


int AlphaZeroPlayer::getAction(std::shared_ptr<Game> game){
	int action;

	std::shared_ptr<GameState> fakeparentparent;
	std::shared_ptr<GameState> fakeparent = std::make_shared<GameState>(game, 0, fakeparentparent);
	std::shared_ptr<GameState> root = std::make_shared<GameState>(game, 0, fakeparent);

	ArrayXf p = this->mcts.simulate(root, this->nn, 1, 200);
	std::cout<<"probablities"<< "\n" << p << std::endl;

	p.maxCoeff(&action);
	
	return action; 
}


int RandomPlayer::getAction(std::shared_ptr<Game> game){
	std::random_device rd;
    std::mt19937 gen(rd());
    std::cout<< "random" << std::endl;
    ArrayXf poss = game->getPossibleActions();
    poss = poss/poss.sum();
    std::discrete_distribution<> dist(poss.data(),poss.data() +  poss.size());

    return dist(gen);
} 
=