#include "NNWrapper.h"
#include "MCTS.cc"
#include <Game.h>
#include <random>
#include "json.hpp"
#include <experimental/filesystem>
#include <ctime>

using json = nlohmann::json;
namespace fs = std::experimental::filesystem;

void save_game(std::shared_ptr<Game> game, std::vector<std::vector<float>> probabilities, std::vector<std::vector<std::vector<float>>> history){
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

void play_game(std::shared_ptr<Game> n_game, MCTS mcts, NNWrapper model, int tempthreshold = 6, bool print = false){
	std::shared_ptr<Game> game = n_game->copy();
	std::vector<std::vector<float>> probabilities;
	std::vector<std::vector<std::vector<float>>> history;

	std::random_device rd;
    std::mt19937 gen(rd());
    

	int action;
	ArrayXf p;
	std::shared_ptr<GameState> fakeparentparent;
	std::shared_ptr<GameState> fakeparent = std::make_shared<GameState>(game, 0, fakeparentparent);
	int game_length = 0;
	float temp = 1;
	while (not game->ended()){
		//save board
    	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> board(game->getBoard());

    	std::vector<std::vector<float>> b_v;
    	for (int i=0; i<board.rows(); ++i){
    		b_v.push_back(std::vector<float>(board.row(i).data(), board.row(i).data() + board.row(i).size()));
    	}
    	history.push_back(b_v);

    	//simulate
		std::shared_ptr<GameState> root = std::make_shared<GameState>(game, 0, fakeparent);
		p = mcts.simulate(root, model, temp, 25);

		//save probability
    	std::vector<float> p_v(p.data(), p.data() + p.size());
    	probabilities.push_back(p_v);

    	//play
    	std::discrete_distribution<> dist(p.data(),p.data() +  p.size());
    	action = dist(gen);
    	game->play(action);
    	game_length++;
    	
    	if (game_length > tempthreshold){
    		temp = 0.1;
    	}

    	if (print){
			game->printBoard();
    	}
	}

	save_game(game, probabilities, history);
}

int main(int argc, char** argv){
	std::shared_ptr<Game> g = Game::create(argv[1]);
	
	MCTS m = MCTS(1.5, 1.10);
	NNWrapper model = NNWrapper(argv[2]);
	int i = 0;
	int RELOAD_MODEL = 3;
	int n_games = std::stoi(argv[3]);

	#pragma omp parallel
	{
		while (true){
			play_game(g, m, model);
			std::cout<<"Game Generated" << std::endl;

			if ((i % RELOAD_MODEL == 0) && (i != 0) ){
				model.reload("models/traced_model_new.pt");
				std::cout<<"Model Updated" << std::endl;
			} 
			
			i++;

			if (n_games > 0 && i > n_games){
				break;
			}
		}
	}
	return 0;
}
