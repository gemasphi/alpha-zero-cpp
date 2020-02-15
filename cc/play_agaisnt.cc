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
		

	    MCTS::Config mcts;

	    Config(cxxopts::ParseResult result) : mcts(result){
			this->id = result["id"].as<std::string>();
  			this->n_games = result["n_games"].as<int>();
  			this->game_name = result["game"].as<std::string>();
  			this->model_loc1 = result["model_one"].as<std::string>();
  			this->model_loc2 = result["model_two"].as<std::string>();
		}

		static void addCommandLineOptions(cxxopts::Options&  options){
			options.add_options()
			  	("i,id", "Id for this test",  cxxopts::value<std::string>())
  				("n,n_games", "Number of games",  cxxopts::value<int>())
  				("g,game", "Game",  cxxopts::value<std::string>())
  				("model_one", "Model Location for player one",  cxxopts::value<std::string>())
  				("model_two", "Model Location for player two",  cxxopts::value<std::string>())
			;

			MCTS::Config::addCommandLineOptions(options);
		}
	};

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


	struct Result {
		std::vector<std::vector<std::vector<float>>> history;
		std::vector<bool> agreement1;
		std::vector<bool> agreement2;
		std::string p1;
		std::string p2;
		int winner;

		Result(Match::Info m): 
				p1(m.p1->name()), p2(m.p2->name()) {};


		void addBoard(MatrixXf game_b){
			Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> board(game_b);
			std::vector<std::vector<float>> b_v;
	    	for (int i=0; i<board.rows(); ++i){
	    		b_v.push_back(std::vector<float>(board.row(i).data(), board.row(i).data() + board.row(i).size()));
	    	}
	    	this->history.push_back(b_v);
		}

		void addAgreement(std::vector<int> best_actions, int action, bool p1){
			bool agree = std::find(best_actions.begin(), best_actions.end(), action) != best_actions.end();
			p1  ? this->agreement1.push_back(agree) 
				: this->agreement2.push_back(agree);
		}

		void setWinner(int winner){
			this->winner = winner;
		}

	};

	struct Player
	{
		std::string name;
		int agreement_count = 0;
		int total_agreement = 0;
		int won = 0;

		Player(std::string name) : name(name) {}	

		void updateAgreement(std::vector<bool> agreement){
			agreement_count += std::count(agreement.begin(), agreement.end(), true);
			total_agreement += agreement.size();
		}
	};



	struct Results
	{
		std::vector<Match::Result> results;
		Match::Config cfg;
		Player p1;
		Player p2;


		Results(Match::Config cfg, Match::Info m): 
				cfg(cfg), p1(m.p1->name()), p2(m.p2->name()) {};

		void addResult(Match::Result res, bool aligned){
			#pragma omp critical
			{
				results.push_back(res);
				p1.updateAgreement(res.agreement1);
				p2.updateAgreement(res.agreement2);

				if(res.winner != 0){
					if(aligned){
						res.winner == 1 ? this->p1.won++ : this->p2.won++;
					} else {
						res.winner == 1 ? this->p2.won++ : this->p1.won++;
					}
				}
			}
		}	
	};

	void to_json(json& j, const Player& p){
		j = json{
				{"name", p.name}, 
				{"won", p.won}, 
			}; 

		if (p.total_agreement != 0){
			j["move_agreement"] =  (float) p.agreement_count / p.total_agreement; 		
		}
	}

	void add_agreement_to_json(json& j, std::string name, std::vector<bool> agreement){
		if (!agreement.empty()){ 
			j[name] = agreement; 
			j[name + "%"] = (float) std::count(agreement.begin(), agreement.end(), true)
							   / agreement.size();
		}
	}

	void to_json(json& j, const Result& r){
		j = json{
				{"winner", r.winner}, 
				{"history", r.history}, 
				{"p1", r.p1}, 
				{"p2", r.p2}, 
			}; 

		add_agreement_to_json(j, "agreement1", r.agreement1);
		add_agreement_to_json(j, "agreement2", r.agreement2);
	}

	void to_json(json& j, const Results& r){
		j = json{
				{"id", r.cfg.id}, 
				{"results", r.results}, 
				{"p1", r.p1}, 
				{"p2", r.p2}, 
				{"draws", r.cfg.n_games - r.p1.won - r.p2.won}, 
			}; 
	}
}


void save_matches(
	Match::Results results,
	std::string type,
	std::string directory = "./temp/playagaisnt_games/"
	){
	directory = directory + "/" + type + "/";

	std::time_t t = std::time(0); 

	if (!fs::exists(directory)){
		std::cout
			<< "Directory " 
			<< directory 
			<< " didn't exist. Creating it..." 
			<< std::endl;
		fs::create_directories(directory);
	}

	json matches = json{results};
	std::ofstream o(directory + std::to_string(t));
	o << matches.dump() << std::endl;
}


Match::Result play_game(Match::Info m, bool print = false){
	std::shared_ptr<Game> game = m.game.copy();
	Player& p1 = *(m.p1);
	Player& p2 = *(m.p2);
	std::shared_ptr<PerfectPlayer> perfectPlayer = m.perfectPlayer;

	int i = 0;
	int action;

	Match::Result result = Match::Result(m);

	while (not game->ended()){

		action = (i % 2 == 0) ? p1.getAction(game)
							  : p2.getAction(game);  


		if (perfectPlayer) result.addAgreement(
							perfectPlayer->getBestActions(game), 
							action, 
							!(m.aligned) == !(i % 2 == 0));
		
		game->play(action);
		result.addBoard(game->getBoard());

		i++;

		if (print) game->printBoard();
	}

	//game->printBoard();
	result.setWinner(game->getWinner());

	return result;
}


void player_vs_player(Match::Config cfg, Match::Info match_b){
	Match::Results results(cfg, match_b);
	int n_threads = cfg.n_games < omp_get_max_threads() ? 1 : omp_get_max_threads();
	
	#pragma omp parallel num_threads(n_threads)
	{
		Match::Info match = match_b;
		for(int i = 0; i < cfg.n_games / n_threads; i++){
			Match::Result result = play_game(match, (cfg.n_games == 1));
			
			std::cout<< "P1-" << match.p1->name() 
					<< " vs" 
					<< " P2-"<< match.p2->name()
					<< ".Winner: " << result.winner << std::endl; 
		
			results.addResult(result, match.aligned);
			match.alternatePlayer();
		}
	}

	save_matches(results, cfg.id);
}

Match::Config parseCommandLine(int argc, char** argv){
	cxxopts::Options options("Player vs Player", "");
	Match::Config::addCommandLineOptions(options); 

  	cxxopts::ParseResult result = options.parse(argc, argv);
  	Match::Config cfg(result);

	return cfg;
}


int main(int argc, char** argv){
	Match::Config cfg = parseCommandLine(argc, argv);
	std::shared_ptr<Game> game = Game::create(cfg.game_name);

	NNWrapper nn1 =  NNWrapper(cfg.model_loc1);
	NNWrapper nn2 =  NNWrapper(cfg.model_loc2);
	std::shared_ptr<AlphaZeroPlayer> p1 = std::make_shared<AlphaZeroPlayer>(nn1, cfg.mcts, 0);
	std::shared_ptr<AlphaZeroPlayer> p2 = std::make_shared<AlphaZeroPlayer>(nn2, cfg.mcts, 0);
	//std::shared_ptr<RandomPlayer> p2 = std::make_shared<RandomPlayer>();
	
	std::shared_ptr<PerfectPlayer> perfectPlayer;

	Match::Info match = Match::Info(
		*game,
		p1,
		p2,
		perfectPlayer);
	
	cfg.id = "vs";
	player_vs_player(cfg, match);

	/*
	perfectPlayer = std::make_shared<ConnectSolver>(""); 
	std::shared_ptr<RandomPlayer> randomPlayer = std::make_shared<RandomPlayer>();
	
	Match::Info pmatch = Match::Info(
		*game,
		p1,
		randomPlayer,
		perfectPlayer);

	cfg.id = "agreement";
	player_vs_player(cfg, pmatch);
	*/
	return 0;
}
