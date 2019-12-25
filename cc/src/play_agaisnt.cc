#include "MCTS.cc"
#include "NNWrapper.h"
#include "Player.h"
#include <Game.h>
#include <Tictactoe.h>
#include <ConnectFour.h>
#include "json.hpp"
#include <experimental/filesystem>


using json = nlohmann::json;
namespace fs = std::experimental::filesystem;

void save_game(std::shared_ptr<Game> game, std::vector<std::vector<std::vector<float>>> history){
	std::string directory = "./playagaisnt_games/";
	std::time_t t = std::time(0); 

	if (!fs::exists(directory)){
		fs::create_directories(directory);
	}

	std::ofstream o(directory + "game_" + std::to_string(t));
	json jgame;
	jgame["winner"] = game->getCanonicalWinner();
	jgame["history"] = history;

	o << jgame.dump() << std::endl;
}

int play_game(Game& n_game, 
				Player& p1, 
				Player& p2, 
				bool print = false){

	std::shared_ptr<Game> game = n_game.copy();
	std::vector<std::vector<std::vector<float>>> history;
	
	int i = 0;
	int action;

	RandomPlayer prand = RandomPlayer();

	while (not game->ended()){
		if (i < 2){
			action = prand.getAction(game);
		}
		else{
			if  (i % 2 == 0){
				action = p1.getAction(game);
			}
			else{
				action = p2.getAction(game);
			}
		}

		i++;
		game->play(action);

		if (print){
			game->printBoard();
		}

		//save board
    	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> board(game->getBoard());

    	std::vector<std::vector<float>> b_v;
    	for (int i=0; i<board.rows(); ++i){
    		b_v.push_back(std::vector<float>(board.row(i).data(), board.row(i).data() + board.row(i).size()));
    	}
    	history.push_back(b_v);
	}

	game->printBoard();
	return game->getWinner();
}

void player_vs_player(Game &game, Player& p1, Player& p2, int n_games = 1){
	int p1_wins = 0;
	int draws = 0;
	int winner;
	for(int i = 0; i < n_games; i++){
		winner = play_game(game, p1, p2, (n_games == 1));

		std::cout<< "One game played, winner: " << winner << std::endl;  
		winner == 0 ? draws++ : ( winner == 1 ? p1_wins++ : i);

	}

	std::cout
	<< "P1 won: " << p1_wins 
	<< " drew: " << draws 
	<< " in " << n_games 
	<< std::endl; 
}

int main(int argc, char** argv){
	std::shared_ptr<Game> g = Game::create(argv[1]);

	MCTS mcts = MCTS(1.5, 1);
	MCTS mcts2 = MCTS(1.5, 1);
	
	NNWrapper nn = NNWrapper(argv[2]);
	//NNWrapper nn2 = NNWrapper(argv[4]);
 
	//AlphaZeroPlayer p1 = AlphaZeroPlayer(nn, mcts);
	//AlphaZeroPlayer p2 = AlphaZeroPlayer(nn2, mcts2);
	//RandomPlayer p2 = RandomPlayer();
	//RandomPlayer p2 = RandomPlayer();
	HumanPlayer p1 = HumanPlayer();
	//HumanPlayer p2 = HumanPlayer();
	ConnectSolver p2 = ConnectSolver(argv[4]);
	
	player_vs_player(*g, p1, p2, std::stoi(argv[3]));

	return 0;
}
