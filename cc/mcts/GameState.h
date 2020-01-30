#ifndef GAMESTATE_H_
#define GAMESTATE_H_

#include <iostream>
#include <map>
#include <Game.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <memory>
#include <limits>  
#include <random>     
using namespace Eigen;

class GameState : public std::enable_shared_from_this<GameState>
{
	private:
		std::vector<std::shared_ptr<GameState>> children;
		int action;
		std::shared_ptr<GameState> parent;
		std::shared_ptr<Game> game;
		bool isExpanded = false;
		ArrayXf childW;
		ArrayXf childP;
		ArrayXf childN;

		void updateW(float v);
		void updateN(int n);
		float getN();
		ArrayXf childQ();
		ArrayXf childU(float cpuct);
		ArrayXf getValidActions(ArrayXf pred, ArrayXf poss);
		int getBestAction(float cpuct);
		std::shared_ptr<GameState> play(int action);
		ArrayXf dirichlet_distribution(ArrayXf alpha);
		
	public:
		GameState(std::shared_ptr<Game> game, int action, std::shared_ptr<GameState> parent);
		GameState(std::shared_ptr<Game> game); //root node constructor
		
		friend std::ostream& operator<<(std::ostream& os, const GameState& gs);
		
		std::shared_ptr<GameState> select(float cpuct);
		void expand(ArrayXf p, float dirichlet_alpha);
		void backup(float v, int n = 1);
		ArrayXf getSearchPolicy(float temp);
		
		//these call the game directly
		bool endGame();
		bool getPlayer();
		bool getWinner();

		std::vector<MatrixXf> getNetworkInput();
		MatrixXf getCanonicalBoard();

		void addVirtualLoss(int vloss);
		void removeVirtualLoss(int vloss);

		std::shared_ptr<GameState> getChild(int action);
};

#endif 
