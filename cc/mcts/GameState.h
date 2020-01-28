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

//TODO add asserts bruv
class GameState : public std::enable_shared_from_this<GameState>
{
	private:
		std::shared_ptr<Game> game;
		int action;
		bool isExpanded;
		std::vector<std::shared_ptr<GameState>> children;

	public:
		std::shared_ptr<GameState> parent;
		ArrayXf childW;
		ArrayXf childP;
		ArrayXf childN;

		GameState(std::shared_ptr<Game> game, int action, std::shared_ptr<GameState> parent);
		
		void addVirtualLoss();
		void removeVirtualLoss();
		
		std::shared_ptr<GameState> select(int cpuct);
		void expand(ArrayXf p, float dirichlet_alpha);
		void backup(float v);
		
		ArrayXf getchildW();
		void updateW(float v);
		void incN();
		float getN();
		float getP();
		ArrayXf childQ();
		ArrayXf childU(float cpuct);
		
		ArrayXf getValidActions(ArrayXf pred, ArrayXf poss);
		ArrayXf getSearchPolicy(float temp);
		
		int getBestAction(ArrayXf puct, std::shared_ptr<GameState> current);

		std::shared_ptr<GameState> getParent();
		
		//these call the game directly
		bool endGame();
		bool getPlayer();
		bool getWinner();


		ArrayXf dirichlet_distribution(ArrayXf alpha);
		MatrixXf getCanonicalBoard();

		std::shared_ptr<GameState> play(int action);
		
		//Todo: this shouldnt belong to this class
		std::vector<MatrixXf> getNetworkInput();

/*		std::shared_ptr<GameState> getChildGameState(int action){
		//	if (this->children.find(action) == this->children.end()){
		}:*/
};

#endif 
