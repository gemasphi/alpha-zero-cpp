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
		int player = 1;
		int winner;
		MatrixXi board;

	public:
		TicTacToe(int boardSize, int player);
		TicTacToe(TicTacToe& t);
		void printBoard();
		void play(int action);
		bool ended();
		int getWinner();
		int getBoardSize();
		int getPlayer();
		bool findWin(Matrix<bool,Dynamic,Dynamic> playerPositions);
		bool isWin(Matrix<bool,Dynamic,Dynamic> smallBoard);
		int getActionSize();
		ArrayXf getPossibleActions();
		MatrixXi getBoard();
};

#endif 