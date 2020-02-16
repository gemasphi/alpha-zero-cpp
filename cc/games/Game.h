#ifndef GAME_H_
#define GAME_H_

#include <iostream>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <memory>
#include <vector>

using namespace Eigen;

class Game
{
	public:
		virtual std::unique_ptr<Game> copy() = 0;
		virtual std::vector<int> getBoardSize() = 0;
		virtual int getActionSize() = 0;
		virtual int getInputPlanes() = 0;
		virtual int getOutputPlanes() = 0;

		virtual void printBoard() = 0;
		virtual void play(int action) = 0;
		virtual bool ended() = 0;
		virtual int getWinner() = 0;
		virtual float getPlayer() = 0;
		virtual ArrayXf getPossibleActions() = 0;
		virtual MatrixXf getBoard() = 0;
		static std::unique_ptr<Game> create(std::string type);
};

#endif 