#include "NNWrapper.h"
#include "MCTS.cc"
#include <Game.h>
#include <Tictactoe.h>
#include <random>
#include "json.hpp"
#include <experimental/filesystem>
#include <ctime>

using json = nlohmann::json;
namespace fs = std::experimental::filesystem;

void save_game(std::shared_ptr<Game> game, std::vector<std::vector<float>> probabilities, std::vector<std::vector<float>> history){
	std::string directory = "./games/";
	std::time_t t = std::time(0); 

	if (!fs::exists(directory)){
		fs::create_directories(directory);
	}

	std::ofstream o(directory + "game_" + std::to_string(t));
	json jgame;
	jgame["probabilities"] = probabilities;
	jgame["winner"] = game->getCanonicalWinner();
	jgame["history"] = history;

	o << jgame.dump() << std::endl;
}

void play_game(std::shared_ptr<Game> n_game, MCTS mcts, NNWrapper model, bool print = false){
	std::shared_ptr<Game> game = n_game->copy();
	std::vector<std::vector<float>> probabilities;
	std::vector<std::vector<float>> history;

	int action;
	std::random_device rd;
    std::mt19937 gen(rd());


	ArrayXf p;
	std::shared_ptr<GameState> fakeparentparent;
	std::shared_ptr<GameState> fakeparent = std::make_shared<GameState>(game, 0, fakeparentparent);
	
	while (not game->ended()){
		MatrixXf b = game->getBoard();
    	std::vector<float> b_v(b.data(), b.data() + b.size());
    	history.push_back(b_v);

		std::shared_ptr<GameState> root = std::make_shared<GameState>(game, 0, fakeparent);
		p = mcts.simulate(root, model, 1, 5);
    	std::discrete_distribution<> dist(p.data(),p.data() +  p.size());
    	action = dist(gen);
    	game->play(action);


    	std::vector<float> p_v(p.data(), p.data() + p.size());
    	probabilities.push_back(p_v);

    
    	
    	if (print){
			game->printBoard();
    	}
	}

	save_game(game, probabilities, history);
}


int main(){
	std::shared_ptr<TicTacToe> t = std::make_shared<TicTacToe>(3,1);
	MCTS m = MCTS(3, 0.3);
	NNWrapper model = NNWrapper("../traced_model.pt");
	
	#pragma omp parallel
	while (true){
		play_game(t, m, model);
		model.reload("../traced_model.pt");
		std::cout<<"One game generated, model reloaded" << std::endl;
	}

	return 0;
}
