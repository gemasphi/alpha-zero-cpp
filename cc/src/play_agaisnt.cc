#include "MCTS.cc"
#include "NNWrapper.h"
#include "Player.h"
#include <Game.h>
#include <Tictactoe.h>


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

int main(){
	TicTacToe t = TicTacToe(3,1);
	MCTS mcts = MCTS(3, 0.3);
	NNWrapper nn = NNWrapper("../traced_model.pt");
	
	AlphaZeroPlayer p1 = AlphaZeroPlayer(nn, mcts);
	HumanPlayer p2 = HumanPlayer();
	play_game(t, p1, p2, true);
	return 0;
}
