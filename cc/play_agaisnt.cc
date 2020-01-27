#include <MCTS.h>
#include <NNWrapper.h>
#include <Player.h>
#include <Game.h>
#include <json.hpp>
#include <experimental/filesystem>
#include <algorithm>

using json = nlohmann::json;
namespace fs = std::experimental::filesystem;

namespace Match{
	struct Info {
		Game& game;
		ProbabilisticPlayer& p1;
		ProbabilisticPlayer& p2;
		std::shared_ptr<PerfectPlayer> perfectPlayer;

		Info(
			Game& game,
			ProbabilisticPlayer& p1,
			ProbabilisticPlayer& p2,
			std::shared_ptr<PerfectPlayer> perfectPlayer
			): 
			game(game),
			p1(p1),
			p2(p2),
			perfectPlayer(perfectPlayer)
			{} 
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


Match::Result play_game(Match::Info m, bool print = false){
	std::shared_ptr<Game> game = m.game.copy();
	ProbabilisticPlayer& p1 = m.p1;
	ProbabilisticPlayer& p2 = m.p2;
	std::shared_ptr<PerfectPlayer> perfectPlayer = m.perfectPlayer;

	int i = 0;
	int action;

	//RandomPlayer prand = RandomPlayer();

	Match::Result result = Match::Result();
	while (not game->ended()){
		action = (i % 2 == 0) ? p1.getAction(game, (i > 3))
							  : p2.getAction(game, (i > 3));  
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

void player_vs_player(std::string id, Match::Info m, int n_games = 1){
	int p1_wins = 0;
	int draws = 0;
	std::vector<Match::Result> results;

	for(int i = 0; i < n_games; i++){
		Match::Result result;
		result = play_game(m, (n_games == 1));
		results.push_back(result);

		std::cout<< "One game played, winner: " << result.winner << std::endl;  

		result.winner == 0 ? draws++ : ( result.winner == 1 ? p1_wins++ : i);
	}

	save_matches(results, id, p1_wins, draws, n_games - p1_wins - draws);
}

int main(int argc, char** argv){
	std::shared_ptr<Game> game = Game::create(argv[1]);
	
	MCTS mcts = MCTS(1.5, 1);
	NNWrapper nn = NNWrapper(argv[2]);
	AlphaZeroPlayer p1 = AlphaZeroPlayer(nn, mcts);
	
	MCTS mcts2 = MCTS(1.5, 1);
	NNWrapper nn2 = NNWrapper(argv[5]);
	AlphaZeroPlayer p2 = AlphaZeroPlayer(nn2, mcts2);
	//NNPlayer p1 = NNPlayer(nn);
	//NNPlayer p2 = NNPlayer(nn2);
	//HumanPlayer p1 = HumanPlayer();
	//HumanPlayer p2 = HumanPlayer();
	//ConnectSolver p1 = ConnectSolver(argv[4]);
	//ConnectSolver p1 = ConnectSolver(argv[4]);
	//ConnectSolver p1 = ConnectSolver(argv[4]);
	
	std::shared_ptr<PerfectPlayer> 
		perfectPlayer;// = std::make_shared<ConnectSolver>(argv[4]); 
	
	//RandomPlayer p2 = RandomPlayer();
	//RandomPlayer p1 = RandomPlayer();

	Match::Info m = Match::Info(
		*game,
		p1,
		p2,
		perfectPlayer);
	
	std::string id = argv[6];
	player_vs_player(id, m, std::stoi(argv[3]));

	return 0;
}
