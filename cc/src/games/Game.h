<<<<<<< HEAD
#ifndef GAME_H_
#define GAME_H_

#include <iostream>
#include <iostream>
#include "../eigen/Eigen/Dense"
#include "../eigen/Eigen/Core"
#include <memory>

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
		virtual int getCanonicalWinner() = 0;
		virtual float getPlayer() = 0;
		virtual ArrayXf getPossibleActions() = 0;
		virtual MatrixXf getBoard() = 0;
		static std::unique_ptr<Game> create(std::string type);
};

=======
#ifndef GAME_H_
#define GAME_H_

#include <iostream>
#include <iostream>
#include "../eigen/Eigen/Dense"
#include "../eigen/Eigen/Core"
#include <memory>

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
		virtual int getCanonicalWinner() = 0;
		virtual float getPlayer() = 0;
		virtual ArrayXf getPossibleActions() = 0;
		virtual MatrixXf getBoard() = 0;
		static std::unique_ptr<Game> create(std::string type);
};

>>>>>>> 4f6c57b7d58c4c7c93b660eccc6a58695228fb3b
#endif 