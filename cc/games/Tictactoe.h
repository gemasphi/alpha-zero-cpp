#ifndef TICTACTOE_H_
#define TICTACTOE_H_

#include <iostream>
#include "Game.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <vector>
using namespace Eigen;

class TicTacToe : public Game
{
	private:
		int boardSize = 3; 
		int inRow = 3; 
		float player = 1;
		int winner;
		MatrixXf board;

	public:
		TicTacToe();
		std::unique_ptr<Game> copy();
		void printBoard();
		int getInputPlanes();
		int getOutputPlanes();
		void play(int action);
		bool ended();
		int getWinner();
		std::vector<int> getBoardSize();
		float getPlayer();
		bool findWin(Matrix<bool,Dynamic,Dynamic> playerPositions);
		bool isWin(Matrix<bool,Dynamic,Dynamic> smallBoard);
		int getActionSize();
		ArrayXf getPossibleActions();
		MatrixXf getBoard();
		int getCanonicalWinner();
};

#endif 