#include <NNWrapper.h>
#include <MCTS.h>
#include <Player.h>
#include <Game.h>
#include <random>
#include <json.hpp>
#include <experimental/filesystem>
#include <ctime>
#include <chrono>

using json = nlohmann::json;
namespace fs = std::experimental::filesystem;

void save_game(std::shared_ptr<Game> game, std::vector<std::vector<float>> probabilities, std::vector<std::vector<std::vector<float>>> history){
	std::string directory = "./temp/games/";
	std::time_t t = std::time(0); 

	if (!fs::exists(directory)){
		fs::create_directories(directory);
	}
    
    int tid = omp_get_thread_num();
	std::ofstream o(directory + "game_" +  std::to_string(tid) +  "_" + std::to_string(t));
	json jgame;
	jgame["probabilities"] = probabilities;
	jgame["winner"] = game->getCanonicalWinner();
	jgame["history"] = history;

	o << jgame.dump() << std::endl;
}

void play_game(std::shared_ptr<Game> n_game, NNWrapper& model, int count, int tempthreshold = 8, bool print = false){
	std::shared_ptr<Game> game = n_game->copy();
	std::vector<std::vector<float>> probabilities;
	std::vector<std::vector<std::vector<float>>> history;

	std::random_device rd;
    std::mt19937 gen(rd());
    
    MCTS::Config mcts = { 
    	2, //cpuct 
    	1, //dirichlet_alpha
    	5, // n_simulations
    	0.1, //temp
    };

    std::shared_ptr<GameState> gs =  std::make_shared<GameState>(game);

	int action;
	ArrayXf p;
	int game_length = 0;

	while (not game->ended()){
		//save board
    	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> board(game->getBoard());

    	std::vector<std::vector<float>> b_v;
    	for (int i=0; i<board.rows(); ++i){
    		b_v.push_back(std::vector<float>(board.row(i).data(), board.row(i).data() + board.row(i).size()));
    	}
    	history.push_back(b_v);

    	//simulate
		p = MCTS::simulate(gs, model, mcts);

		//save probability
    	std::vector<float> p_v(p.data(), p.data() + p.size());
    	probabilities.push_back(p_v);

    	//playls
    	std::discrete_distribution<> dist(p.data(),p.data() +  p.size());
    	action = dist(gen);
    	game->play(action);
    	game_length++;
    	
    	gs = gs->getChild(action);
    	

    	if (game_length > tempthreshold){
    		mcts.temp = 1.5;
    	}

    	if (print){
			game->printBoard();
    	}
	}

	save_game(game, probabilities, history);
}

void play_perfectly(std::shared_ptr<Game> n_game, Player& perfectPlayer){
	std::shared_ptr<Game> game = n_game->copy();
	std::vector<std::vector<float>> probabilities;
	std::vector<std::vector<std::vector<float>>> history;

	int i;
	int action;

	RandomPlayer r_player = RandomPlayer(); 
	while (not game->ended()){
		bool randomPlay = (i < 1);
 		action = randomPlay ? r_player.getAction(game) : perfectPlayer.getAction(game);

 		if (!randomPlay){
 			Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> board(game->getBoard());

	    	std::vector<std::vector<float>> b_v;
	    	for (int i=0; i<board.rows(); ++i){
	    		b_v.push_back(std::vector<float>(board.row(i).data(), board.row(i).data() + board.row(i).size()));
	    	}
	    	
	    	history.push_back(b_v);

    		std::vector<float> v(game->getActionSize(), 0);
    		v[action] = 1;
			probabilities.push_back(v);
 		}

    	game->play(action);
		i++;
	}

	save_game(game, probabilities, history);
}
int main(int argc, char** argv){
	std::shared_ptr<Game> g = Game::create(argv[1]);
	int RELOAD_MODEL = 3;
	int n_games = std::stoi(argv[3]);

	std::cout << "n games:" << n_games<< std::endl;
//	#pragma omp parallel
	{	
	int i = 0;
	int count = 1;
	NNWrapper model = NNWrapper(argv[2]);
		while (true){
			play_game(g, model, count);
			auto now = std::chrono::system_clock::now();
			std::time_t now_time = std::chrono::system_clock::to_time_t(now);
			
			std::cout<< std::ctime(&now_time) <<" Game Generated" << std::endl;

			
			if ((i % RELOAD_MODEL == 0) && (i != 0) && n_games == -1 ){
				model.reload(argv[2]);
				std::cout<<"Model Updated" << std::endl;
			} 
			
			i++;
			std::cout << "i:" << i<< std::endl;

			count++;
			/*if (n_games > 0 && i == n_games){
				break;
			}*/
		}
	}
	return 0;
}
