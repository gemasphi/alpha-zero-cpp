#ifndef TICTACTOE_H_
#define TICTACTOE_H_

#include <iostream>
#include "eigen/Eigen/Dense"
#include "eigen/Eigen/Core"

using namespace Eigen;

class TicTacToe
{
	private:
		int boardSize = 3; 
		int inRow = 3; 
		float player = 1;
		int winner;
		MatrixXf board;

	public:
		TicTacToe(int boardSize, float player);
		TicTacToe(TicTacToe& t);
		void printBoard();
		void play(int action);
		bool ended();
		int getWinner();
		int getBoardSize();
		float getPlayer();
		bool findWin(Matrix<bool,Dynamic,Dynamic> playerPositions);
		bool isWin(Matrix<bool,Dynamic,Dynamic> smallBoard);
		int getActionSize();
		ArrayXf getPossibleActions();
		MatrixXf getBoard();
};

#endif 