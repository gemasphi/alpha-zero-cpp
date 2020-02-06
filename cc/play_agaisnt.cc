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

namespace Match{
	struct Config{
		std::string id;
		int n_games;
		std::string game_name;
		std::string model_loc1;
		std::string model_loc2;


	    MCTS::Config mcts = { 
	    	2, //cpuct 
	    	1, //dirichlet_alpha
	    	5, // n_simulations
	    	1, //temp
	    };
	};

	struct Info {
		Game& game;
		Player& p1;
		Player& p2;
		std::shared_ptr<PerfectPlayer> perfectPlayer;

		Info(
			Game& game,
			Player& p1,
			Player& p2,
			std::shared_ptr<PerfectPlayer> perfectPlayer
			): 
			game(game),
			p1(p1),
			p2(p2),
			perfectPlayer(perfectPlayer)
			{} 

		void alternatePlayer(){
			Player& _p1 = this->p1;
			this->p1 = this->p2;
			this->p2 = _p1;
		}
	};

	struct Result {
		std::vector<std::vector<std::vector<float>>> history;
		std::vector<bool> agreement1;
		std::vector<bool> agreement2;
		int winner;

		Result(){};

		void addBoard(MatrixXf game_b){
			Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> board(game_b);
			std::vector<std::vector<float>> b_v;
	    	for (int i=0; i<board.rows(); ++i){
	    		b_v.push_back(std::vector<float>(board.row(i).data(), board.row(i).data() + board.row(i).size()));
	    	}
	    	this->history.push_back(b_v);
		}

		void addAgreement(bool agree, bool p1){
			p1  ? this->agreement1.push_back(agree) 
				: this->agreement2.push_back(agree);
		}

		void setWinner(int winner){
			this->winner = winner;
		}

	};

	void to_json(json& j, const Result& r){
		j = json{
				{"winner", r.winner}, 
				{"history", r.history}, 
				{"agreement1", r.agreement1},
				{"agreement2", r.agreement2},
			}; 
	}
}


void save_matches(
	std::vector<Match::Result> results, 
	std::string id,
	int p1_wins, 
	int draws,
	int p2_wins, 
	std::string directory = "./temp/playagaisnt_games/"
	){
	
	std::time_t t = std::time(0); 

	if (!fs::exists(directory)){
		std::cout
			<< "Directory " 
			<< directory 
			<< " didn't exist. Creating it..." 
			<< std::endl;
		fs::create_directories(directory);
	}

	json matches = json{
		{"id", id}, 
		{"results", results}, 
		{"p1_wins", p1_wins}, 
		{"draws", draws}, 
		{"p2_wins", p2_wins}, 
	};

	std::ofstream o(directory + "matches_" + std::to_string(t));
	o << matches.dump() << std::endl;
}


Match::Result play_game(Match::Info m, bool print = false){
	std::shared_ptr<Game> game = m.game.copy();
	Player& p1 = m.p1;
	Player& p2 = m.p2;
	std::shared_ptr<PerfectPlayer> perfectPlayer = m.perfectPlayer;

	int i = 0;
	int action;

	//RandomPlayer prand = RandomPlayer();

	Match::Result result = Match::Result();
	while (not game->ended()){

		action = (i % 2 == 0) ? p1.getAction(game)
							  : p2.getAction(game);  
							  //: p2.getAction(game, (i > 3));  

		if (perfectPlayer and i%2 == 0){
			std::vector<int> best_actions;	
			perfectPlayer->getAction(game, best_actions);
			bool agree = std::find(best_actions.begin(), best_actions.end(), action) != best_actions.end();
			result.addAgreement(agree, (i % 2 == 0));
		}

		i++;
		game->play(action);
		result.addBoard(game->getBoard());

		if (print){
			game->printBoard();
		}
	}

	game->printBoard();
	result.setWinner(game->getWinner());

	return result;
}


void player_vs_player(std::string id, Match::Info match, int n_games = 1){
	int p1_wins = 0;
	int draws = 0;
	std::vector<Match::Result> results;

	for(int i = 0; i < n_games; i++){
		Match::Result result = play_game(match, (n_games == 1));
		results.push_back(result);
		
		std::cout<< "One game played, winner: " << result.winner << std::endl; 
		match.alternatePlayer();

		if(result.winner == 0){
			draws++;	
		}  
		else if(i%2 == 0){
			result.winner == 1 ? p1_wins++ : i;
		}
		else{
		 	result.winner != 1 ? p1_wins++ : i;
		}
	}

	save_matches(results, id, p1_wins, draws, n_games - p1_wins - draws);
}

Match::Config parseCommandLine(int argc, char** argv){
	cxxopts::Options options("Selfplay", "");
	options.add_options()
  		("i,id", "Id for this test",  cxxopts::value<std::string>())
  		("n,n_games", "Number of games",  cxxopts::value<int>())
  		("g,game", "Game",  cxxopts::value<std::string>())
  		("model_one", "Model Location for player one",  cxxopts::value<std::string>())
  		("model_two", "Model Location for player two",  cxxopts::value<std::string>())
  	;
  		//("n_p,n_games_perfect", "Number of games agaisnt perfectPlayer",  cxxopts::value<int>())

  	auto result = options.parse(argc, argv);

  	Match::Config cfg{
  		.id = result["id"].as<std::string>(),
  		.n_games = result["n_games"].as<int>(),
  		.game_name = result["game"].as<std::string>(),
  		.model_loc1 = result["model_one"].as<std::string>(),
  		.model_loc2 = result["model_two"].as<std::string>(),
  	};

	return cfg;
}


int main(int argc, char** argv){
	Match::Config cfg = parseCommandLine(argc, argv);

	std::shared_ptr<Game> game = Game::create(cfg.game_name);

	NNWrapper nn1 =  NNWrapper(cfg.model_loc1);
	NNWrapper nn2 =  NNWrapper(cfg.model_loc2);
	AlphaZeroPlayer p1 = AlphaZeroPlayer(nn1, cfg.mcts, 3);
	AlphaZeroPlayer p2 = AlphaZeroPlayer(nn2, cfg.mcts, 3);
	
	//ConnectSolver p1 = ConnectSolver(argv[4]);
	
	std::shared_ptr<PerfectPlayer> 
		perfectPlayer;// = std::make_shared<ConnectSolver>(argv[4]); 
	
	//RandomPlayer p2 = RandomPlayer();
	//RandomPlayer p1 = RandomPlayer();

	Match::Info match = Match::Info(
		*game,
		p1,
		p2,
		perfectPlayer);
	
	player_vs_player(cfg.id, match, cfg.n_games);

	return 0;
}
