#ifndef PLAYER_H_
#define PLAYER_H_

#include <Game.h>
#include <Solver.hpp>
#include <ConnectFour.h>
#include <NNWrapper.h>
#include <MCTS.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <iostream>

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
	private:
		int howManyMovesPlayed = 0;
		int deterministicAfter;

	public:
		ProbabilisticPlayer(int deterministicAfter);
		int getAction(std::shared_ptr<Game> game);
		virtual ArrayXf getProbabilities(std::shared_ptr<Game> game) = 0;
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
		NNPlayer(NNWrapper& nn, int deterministicAfter);
		ArrayXf getProbabilities(std::shared_ptr<Game> game);

};

class AlphaZeroPlayer : public ProbabilisticPlayer
{	
	private:
		NNWrapper& nn;
		MCTS::Config mcts;
		bool parallel;

	public:
		AlphaZeroPlayer(NNWrapper& nn, MCTS::Config mcts, int deterministicAfter);
		ArrayXf getProbabilities(std::shared_ptr<Game> game);
};

#endif 