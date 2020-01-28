#ifndef CONNECTFOUR_H_
#define CONNECTFOUR_H_

#include <iostream>
#include "Game.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <string>
#include <Position.hpp>

using namespace GameSolver::Connect4;
using namespace Eigen;

class ConnectFour : public Game
{
	private:
		MatrixXf board;
		ArrayXf allowedToPlay;
		int height;
		int width;
		int player;
		int inRow;
		int winner;
		int boardSize;
		std::string played;
		Position board_rep;

	public:
		ConnectFour();
		std::unique_ptr<Game> copy();
		void printBoard();
		void play(int action);
		int getInputPlanes();
		int getOutputPlanes();
		bool ended();
		int getWinner();
		int getEmptyColIndex(int col);
		std::vector<int> getBoardSize();
		float getPlayer();
		int getActionSize();
		ArrayXf getPossibleActions();
		MatrixXf getBoard();
		int getCanonicalWinner();
		std::string getPlayedMoves();
};

#endif 