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

std::string HumanPlayer::name(){
	return "Human Player";
}


int PerfectPlayer::getAction(std::shared_ptr<Game> game){
	return pickRandomElement(this->getBestActions(game));
}

ConnectSolver::ConnectSolver(std::string opening_book) {
  this->solver.loadBook(opening_book);	
}

std::vector<int> ConnectSolver::getBestActions(std::shared_ptr<Game> game){
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

std::string ConnectSolver::name(){
	return "Perfect C4 Player";
}

ProbabilisticPlayer::ProbabilisticPlayer(int deterministicAfter) : deterministicAfter(deterministicAfter) {}

int ProbabilisticPlayer::getAction(std::shared_ptr<Game> game){
	ArrayXf p = this->getProbabilities(game);
	int action;
	//std::cout<< p<<"\n fds" << std::endl;
	if (this->howManyMovesPlayed > this->deterministicAfter){
		p.maxCoeff(&action);
	} else {
		action = pickStochasticElement(p);
	}

	this->howManyMovesPlayed++;

	return action;
}

AlphaZeroPlayer::AlphaZeroPlayer(NNWrapper& nn, MCTS::Config mcts, int deterministicAfter): 
								ProbabilisticPlayer(deterministicAfter),
								nn(nn), mcts(mcts){}

ArrayXf AlphaZeroPlayer::getProbabilities(std::shared_ptr<Game> game){
	return MCTS::simulate(game, this->nn, this->mcts); 
}

std::string AlphaZeroPlayer::name(){
	return "AZ Player: " + this->nn.getFilename();
}

MCTSPlayer::MCTSPlayer(MCTS::Config mcts, int deterministicAfter): 
								ProbabilisticPlayer(deterministicAfter),
								mcts(mcts){}

ArrayXf MCTSPlayer::getProbabilities(std::shared_ptr<Game> game){
	return MCTS::simulate_random(game, this->mcts); 
}

std::string MCTSPlayer::name(){
	return "MCTS Player";
}

NNPlayer::NNPlayer(NNWrapper& nn, int deterministicAfter) : ProbabilisticPlayer(deterministicAfter), nn(nn) {}

ArrayXf NNPlayer::getProbabilities(std::shared_ptr<Game> game){
	NN::Input i = NN::Input({game->getBoard()*game->getPlayer()});
	NN::Output res = this->nn.predict(i)[0];

	ArrayXf poss = game->getPossibleActions();
	ArrayXf valid_actions = poss*res.policy;
	

	return valid_actions;
}


std::string NNPlayer::name(){
	return "NN Player: " + this->nn.getFilename();
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

std::string RandomPlayer::name(){
	return "Randomy Player";
}
