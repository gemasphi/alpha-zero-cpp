#ifndef PLAYER_H_
#define PLAYER_H_

#include "games/Game.h"
#include "games/ConnectFour.h"
#include "NNWrapper.h"
#include "MCTS.cc"
#include "../eigen/Eigen/Dense"
#include "../eigen/Eigen/Core"
#include <iostream>
#include "games/connect4solver/Solver.hpp"

using namespace GameSolver::Connect4;


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

class ConnectSolver : public Player
{
	Solver solver;
	public:
		ConnectSolver(std::string opening_book);
		int getAction(std::shared_ptr<Game> game);
};

/*
class NNPlayer : public Player
{
	private:
		NNWrapper nn;

	public:
		NNPlayer(NNWrapper nn);
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