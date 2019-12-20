#include "MCTS.cc"
#include "NNWrapper.h"
#include "Player.h"
#include <Game.h>
#include <Tictactoe.h>
#include <ConnectFour.h>

int play_game(Game& n_game, 
				Player& p1, 
				Player& p2, 
				bool print = false){

	std::shared_ptr<Game> game = n_game.copy();
	int i = 0;
	int action;

	while (not game->ended()){
		Player& current_player = (i % 2 == 0) ? p1 : p2;
		i++;
		
		action = current_player.getAction(game);
		game->play(action);

		if (print){
			game->printBoard();
		}
	}

	return game->getCanonicalWinner();
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
	MCTS mcts = MCTS(2.5, 1.10);
	NNWrapper nn = NNWrapper(argv[2]);

	AlphaZeroPlayer p1 = AlphaZeroPlayer(nn, mcts);
	//RandomPlayer p2 = RandomPlayer();
	HumanPlayer p2 = HumanPlayer();
	//HumanPlayer p1 = HumanPlayer();
	
	player_vs_player(*g, p1, p2, std::stoi(argv[3]));

	return 0;
}
