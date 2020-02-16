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
		virtual std::string name() = 0;
};

class PerfectPlayer : public Player
{
	public:
		int getAction(std::shared_ptr<Game> game);
		virtual std::vector<int> getBestActions(std::shared_ptr<Game> game) = 0;

};

class ConnectSolver : public PerfectPlayer
{
	Solver solver;
	public:
		ConnectSolver(std::string opening_book);
		std::vector<int> getBestActions(std::shared_ptr<Game> game);
		std::string name();

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
		std::string name();
};

class RandomPlayer : public Player
{
	public:
		int getAction(std::shared_ptr<Game> game);
		std::string name();
};


class NNPlayer : public ProbabilisticPlayer
{
	NNWrapper& nn;
	
	public:
		NNPlayer(NNWrapper& nn, int deterministicAfter);
		ArrayXf getProbabilities(std::shared_ptr<Game> game);
		std::string name();

};

class AlphaZeroPlayer : public ProbabilisticPlayer
{	
	private:
		NNWrapper& nn;
		MCTS::Config mcts;

	public:
		AlphaZeroPlayer(NNWrapper& nn, MCTS::Config mcts, int deterministicAfter);
		ArrayXf getProbabilities(std::shared_ptr<Game> game);
		std::string name();
};

class MCTSPlayer : public ProbabilisticPlayer
{	
	private:
		MCTS::Config mcts;

	public:
		MCTSPlayer(MCTS::Config mcts, int deterministicAfter);
		ArrayXf getProbabilities(std::shared_ptr<Game> game);
		std::string name();
};

#endif 