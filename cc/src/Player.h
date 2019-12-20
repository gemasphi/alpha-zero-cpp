#ifndef PLAYER_H_
#define PLAYER_H_

#include "games/Game.h"
#include "NNWrapper.h"
#include "MCTS.cc"
#include "../eigen/Eigen/Dense"
#include "../eigen/Eigen/Core"
#include <iostream>

class Player
{
	public:
		virtual int getAction(std::shared_ptr<Game> game) = 0;
		static std::unique_ptr<Player> create(std::string type);

};


class HumanPlayer : public Player
{
	public:
		int getAction(std::shared_ptr<Game> game);
};

class RandomPlayer : public Player
{
	public:
		int getAction(std::shared_ptr<Game> game);
};
/*
class NNPlayer : public Player
{
	private:
		NNWrapper nn;

	public:
		int getAction(std::shared_ptr<Game> game);
};
*/
class AlphaZeroPlayer : public Player
{	
	private:
		NNWrapper nn;
		MCTS mcts;

	public:
		AlphaZeroPlayer(NNWrapper nn, MCTS mcts);
		int getAction(std::shared_ptr<Game> game);

};

#endif 