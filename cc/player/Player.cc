#include "Player.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <iostream>
#include <random>
#include <NNWrapper.h>
	
int pickRandomElement(std::vector<int> v){
	std::random_device random_device;   
	std::mt19937 engine{random_device()};   
	std::uniform_int_distribution<int> dist(0, v.size() - 1);

	return  v[dist(engine)]; 
}

int pickStochasticElement(ArrayXf p){
	std::random_device random_device;   
	std::mt19937 engine{random_device()}; 

    std::discrete_distribution<> dist(p.data(),p.data() +  p.size());

	return  dist(engine); 
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


int ConnectSolver::getAction(std::shared_ptr<Game> game, std::vector<int>& best_indexes){
	best_indexes = this->calcScores(game);
	return pickRandomElement(best_indexes); 
}

int ConnectSolver::getAction(std::shared_ptr<Game> game){
	return pickRandomElement(this->calcScores(game));
}


std::vector<int> ConnectSolver::calcScores(std::shared_ptr<Game> game){
	std::shared_ptr<ConnectFour> c_game = std::dynamic_pointer_cast<ConnectFour>(game); 
    ArrayXf poss = c_game->getPossibleActions();
	
	int max_possible_score = c_game->getBoardSize()[0]*c_game->getBoardSize()[1];
	int max_score = max_possible_score*-1;
	std::vector<int> max_index;
	int score;

	for (int i = 0; i < poss.size(); i++){
		if (poss[i] != 0){
			Position pos;
			std::string to_play = c_game->getPlayedMoves()+ std::to_string(i + 1);
    		
    		if (pos.play(to_play) != to_play.size()){
				score = max_possible_score;
    		} else{
	    		score = this->solver.solve(pos, false)*-1;
    		}

			if (score > max_score){
				max_score = score;
				max_index = std::vector<int>();
				max_index.push_back(i);
			} else if(score == max_score){
				max_index.push_back(i);
			}

    	}
    }

    return max_index;
}


AlphaZeroPlayer::AlphaZeroPlayer(NNWrapper& nn, MCTS& mcts): nn(nn), mcts(mcts){}

int AlphaZeroPlayer::getAction(std::shared_ptr<Game> game){
	return this->getAction(game, true); 
}

int AlphaZeroPlayer::getAction(std::shared_ptr<Game> game, bool deterministc){
	int action;

	std::shared_ptr<GameState> fakeparentparent;
	std::shared_ptr<GameState> fakeparent = std::make_shared<GameState>(game, 0, fakeparentparent);
	std::shared_ptr<GameState> root = std::make_shared<GameState>(game, 0, fakeparent);

	ArrayXf p = this->mcts.simulate(root, this->nn, 1, 600);
	//std::cout<<"probablities"<< "\n" << p << std::endl;

	if (deterministc){
		p.maxCoeff(&action);
	} else {
		action = pickStochasticElement(p);
	}
	
	return action; 
}


NNPlayer::NNPlayer(NNWrapper& nn) : nn(nn) {}

int NNPlayer::getAction(std::shared_ptr<Game> game){
	return this->getAction(game, true);
}

int NNPlayer::getAction(std::shared_ptr<Game> game, bool deterministc){
	int action;

	NN::Input i = NN::Input({game->getBoard()*game->getPlayer()});
	NN::Output res = this->nn.predict(i)[0];

	ArrayXf poss = game->getPossibleActions();
	ArrayXf valid_actions = poss*res.policy;
	
	if (deterministc){
		valid_actions.maxCoeff(&action);
	} else {
		action = pickStochasticElement(valid_actions);
	}

	return action;
}

int RandomPlayer::getAction(std::shared_ptr<Game> game){
	std::random_device rd;
    std::mt19937 gen(rd());
    //std::cout<< "random" << std::endl;
    ArrayXf poss = game->getPossibleActions();
    poss = poss/poss.sum();
    std::discrete_distribution<> dist(poss.data(),poss.data() +  poss.size());

    return dist(gen);
} 
