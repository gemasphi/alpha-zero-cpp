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

};

class PerfectPlayer : public Player
{
	public:
		virtual int getAction(std::shared_ptr<Game> game, std::vector<int>& best_scores) = 0;

};

class ProbabilisticPlayer : public Player
{
	public:
		virtual int getAction(std::shared_ptr<Game> game, bool deterministc) = 0;

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



class ConnectSolver : public PerfectPlayer
{
	Solver solver;
	public:
		ConnectSolver(std::string opening_book);
		int getAction(std::shared_ptr<Game> game, std::vector<int>& best_scores);
		int getAction(std::shared_ptr<Game> game);

	private:
		std::vector<int> calcScores(std::shared_ptr<Game> game);
};

class NNPlayer : public ProbabilisticPlayer
{
	NNWrapper& nn;
	
	public:
		NNPlayer(NNWrapper& nn);
		int getAction(std::shared_ptr<Game> game);
		int getAction(std::shared_ptr<Game> game, bool deterministc);

};

class AlphaZeroPlayer : public ProbabilisticPlayer
{	
	private:
		NNWrapper& nn;
		MCTS& mcts;


	public:
		AlphaZeroPlayer(NNWrapper& nn, MCTS& mcts);
		int getAction(std::shared_ptr<Game> game);
		int getAction(std::shared_ptr<Game> game, bool deterministc);
};

#endif 