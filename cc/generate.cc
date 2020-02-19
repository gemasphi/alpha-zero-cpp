#include <MCTS.h>
#include <NNWrapper.h>
#include <Player.h>
#include <Game.h>
#include <json.hpp>
#include <experimental/filesystem>
#include <algorithm>
#include <cxxopts.hpp>

using json = nlohmann::json;
namespace fs = std::experimental::filesystem;

namespace Generate{
	struct Info {
		Game& game;
		std::shared_ptr<Player> p1;
		std::shared_ptr<Player> p2;
		std::shared_ptr<PerfectPlayer> perfectPlayer;

		bool aligned = true; //if model_loc1 = p1 etc

		Info(
			Game& game,
			std::shared_ptr<Player> p1,
			std::shared_ptr<Player> p2,
			std::shared_ptr<PerfectPlayer> perfectPlayer
			): 
			game(game),
			p1(p1),
			p2(p2),
			perfectPlayer(perfectPlayer)
			{} 

		void alternatePlayer(){
			p1.swap(p2);
			aligned = !aligned;
		}
	};

	struct Result
	{
		std::vector<std::vector<float>> probabilities;
		std::vector<int> value;
		std::vector<std::vector<std::vector<float>>> history;

		Result(){};

		void addBoardProbs(MatrixXf game_b, std::vector<float> scores){
			this->addBoard(game_b);
			this->addProbabilityValue(scores);
		}

		void addBoard(MatrixXf game_b){
			Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> board(game_b);
			std::vector<std::vector<float>> b_v;
	    	for (int i=0; i<board.rows(); ++i){
	    		b_v.push_back(std::vector<float>(board.row(i).data(), board.row(i).data() + board.row(i).size()));
	    	}
	    	this->history.push_back(b_v);
		}

		void addProbabilityValue(std::vector<float> v_scores){
			float max_score;
   			Map<ArrayXf>  scores(v_scores.data(), v_scores.size());
   			scores.maxCoeff(&max_score);
   			float value = scores[max_score] > 0 ? 1 : -1;

   			//std::cout<< "max score:"<< max_score<<std::endl;
   			//std::cout<< scores<<std::endl;
	    	scores = scores/value;
   			//std::cout<< "after divsion\n"<< scores<<std::endl;
	    	scores = scores.max(0);
   			//std::cout<< "min divsion\n"<< scores<<std::endl;
	    	scores = scores / scores.sum();
   			//std::cout<< "after norm\n"<< scores<<std::endl;

	    	std::vector<float> p(scores.data(), scores.data() + scores.size());
	    	this->probabilities.push_back(p);

	    	this->value.push_back(value);
		}

	};
}

void save_game(Generate::Result res){
	std::string directory = "./temp/perfect_player/";
	std::time_t t = std::time(0); 

	if (!fs::exists(directory)){
		fs::create_directories(directory);
	}
    
	std::ofstream o(directory + std::to_string(omp_get_thread_num()) +  "_" + std::to_string(t));
	json jgame;
	jgame["probabilities"] = res.probabilities;
	jgame["value"] = res.value;
	jgame["history"] = res.history;

	o << jgame.dump() << std::endl;
}


void play_game(Generate::Info m){
	std::shared_ptr<Game> game = m.game.copy();
	Player& p1 = *(m.p1);
	Player& p2 = *(m.p2);
	std::shared_ptr<PerfectPlayer> perfectPlayer = m.perfectPlayer;

	int i = 0;
	int action;

	Generate::Result res = Generate::Result();

	while (not game->ended()){
		action = (i % 2 == 0) ? p1.getAction(game)
							  : p2.getAction(game);  

	   if (perfectPlayer){
	    	std::vector<float> scores = perfectPlayer->getBestScores(game);
	    	res.addBoardProbs(game->getBoard(),scores);
	    }
		game->play(action);
		i++;
	}

	save_game(res);
}

int main(int argc, char** argv){
	std::shared_ptr<Game> game = Game::create("CONNECTFOUR");
	std::shared_ptr<RandomPlayer> p1 = std::make_shared<RandomPlayer>();
	std::shared_ptr<RandomPlayer> p2 = std::make_shared<RandomPlayer>();
	std::shared_ptr<ConnectSolver> perfectPlayer = std::make_shared<ConnectSolver>("_deps/connect4solver-src/7x6.book");

	Generate::Info match = Generate::Info(
		*game,
		p1,
		p2,
		perfectPlayer);
	
	#pragma omp parallel
	{
		Generate::Info thread_match = match;
		while (true){
			play_game(thread_match);
			thread_match.alternatePlayer();

			auto now = std::chrono::system_clock::now();
			std::time_t now_time = std::chrono::system_clock::to_time_t(now);
			std::cout<< std::ctime(&now_time) <<" Game Generated" << std::endl;
		}
	}

	return 0;
}
