#include <NNWrapper.h>
#include <MCTS.h>
#include <Player.h>
#include <Game.h>
#include <random>
#include <json.hpp>
#include <cxxopts.hpp>
#include <experimental/filesystem>
#include <ctime>
#include <chrono>

using json = nlohmann::json;
namespace fs = std::experimental::filesystem;


namespace Selfplay{
	struct Config {
		int n_games;
		std::string game_name;
		std::string model_loc;

		int tempthreshold = 8;
		float afterThresholdTemp = 1.5;
		bool print = false;

		MCTS::Config mcts = { 
    		2, //cpuct 
    		1, //dirichlet_alpha
    		25, // n_simulations
    		0.1, //temp
    	};
	};

	struct Result
	{
		std::vector<std::vector<float>> probabilities;
		std::vector<std::vector<std::vector<float>>> history;

		Result(){};

		void addBoardProbs(MatrixXf game_b, ArrayXf p){
			this->addBoard(game_b);
			this->addProbability(p);
		}

		void addBoard(MatrixXf game_b){
			Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> board(game_b);
			std::vector<std::vector<float>> b_v;
	    	for (int i=0; i<board.rows(); ++i){
	    		b_v.push_back(std::vector<float>(board.row(i).data(), board.row(i).data() + board.row(i).size()));
	    	}
	    	this->history.push_back(b_v);
		}

		void addProbability(ArrayXf p){
	    	std::vector<float> p_v(p.data(), p.data() + p.size());
	    	this->probabilities.push_back(p_v);
		}

	};
}


void save_game(std::shared_ptr<Game> game, Selfplay::Result res){
	std::string directory = "./temp/games/";
	std::time_t t = std::time(0); 

	if (!fs::exists(directory)){
		fs::create_directories(directory);
	}
    
    int tid = omp_get_thread_num();
	std::ofstream o(directory + "game_" +  std::to_string(tid) +  "_" + std::to_string(t));
	json jgame;
	jgame["probabilities"] = res.probabilities;
	jgame["winner"] = game->getCanonicalWinner();
	jgame["history"] = res.history;

	o << jgame.dump() << std::endl;
}

int pickAction(ArrayXf p){
	static std::random_device rd;
    static std::mt19937 gen(rd());
	
	std::discrete_distribution<> dist(p.data(),p.data() +  p.size());
    return dist(gen);
}

void play_game(
	std::shared_ptr<Game> n_game, 
	NNWrapper& model, 
	Selfplay::Config cfg
	){

	std::shared_ptr<Game> game = n_game->copy();
    std::shared_ptr<GameState> gs =  std::make_shared<GameState>(game);
    Selfplay::Result gameResult;

	int action;
	ArrayXf p;
	int game_length = 0;

	while (not game->ended()){
		p = MCTS::simulate(gs, model, cfg.mcts);

		gameResult.addBoardProbs(game->getBoard(), p);
	
    	action = pickAction(p);
    	game->play(action);
    	
    	game_length++;
    	
    	gs = gs->getChild(action);

    	if (game_length > cfg.tempthreshold){
    		cfg.mcts.temp = cfg.afterThresholdTemp;
    	}

    	if (cfg.print){
			game->printBoard();
    	}
	}

	save_game(game, gameResult);
}


Selfplay::Config parseCommandLine(int argc, char** argv){
	cxxopts::Options options("Selfplay", "");
	options.add_options()
  		("m,model", "Model Location",  cxxopts::value<std::string>())
  		("g,game", "Game",  cxxopts::value<std::string>())
  		("n,n_games", "Number of games",  cxxopts::value<int>())
  	;

  	auto result = options.parse(argc, argv);
  	Selfplay::Config cfg{
  		.n_games = result["n_games"].as<int>(),
  		.game_name = result["game"].as<std::string>(),
  		.model_loc = result["model"].as<std::string>(),
  	};

	return cfg;
}

int main(int argc, char** argv){
	Selfplay::Config cfg = parseCommandLine(argc, argv); 

	std::shared_ptr<Game> g = Game::create(cfg.game_name);
	NNWrapper model = NNWrapper(cfg.model_loc);
	int n_games = (cfg.n_games / omp_get_max_threads()) + 1;
	
	#pragma omp parallel
	{	
		int i = 0;
		while (true){
			play_game(g, model, cfg);

			auto now = std::chrono::system_clock::now();
			std::time_t now_time = std::chrono::system_clock::to_time_t(now);
			std::cout<< std::ctime(&now_time) <<" Game Generated" << std::endl;

			#pragma omp critical
			{
				model.shouldLoad(cfg.model_loc);
			}		

			i++;
			if (n_games > 0 && i == n_games){
				break;
			}
		}
	}
	return 0;
}
