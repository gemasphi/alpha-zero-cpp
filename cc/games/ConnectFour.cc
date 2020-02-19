#include "ConnectFour.h"
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <stdexcept>
#include <sstream>
#include <memory>

using namespace Eigen;

ConnectFour::ConnectFour(){
	this->width = Position::WIDTH;
	this->height = Position::HEIGHT;
	this->board = MatrixXf::Zero(this->height, this->width);
	this->player = 1;
	this->inRow = 4;
	this->played = "";
	this->allowedToPlay = ArrayXf::Zero(this->width);
}

std::unique_ptr<Game> ConnectFour::copy(){
	return std::make_unique<ConnectFour>(*this);
}

int ConnectFour::getInputPlanes(){
	return 5;
}

int ConnectFour::getOutputPlanes(){
	return 1;
}

int ConnectFour::getActionSize(){
	return this->width;
}

std::vector<int> ConnectFour::getBoardSize(){
	return {this->height, this->width};
}

std::string ConnectFour::getPlayedMoves(){
	return this->played;
}

void ConnectFour::printBoard(){
	std::cout<< this->board << std::endl;
}

void ConnectFour::play(int action){
	int indexToPlay = this->getEmptyColIndex(action);

	if (indexToPlay == -1){
		std::ostringstream  error;
		error << "Invalid action: " << action << "\n" << this->board; 
		throw std::invalid_argument(error.str());
	}


	this->board(indexToPlay, action) = this->player;
	this->allowedToPlay[action]++;
	this->board_rep.playCol(action);
	this->played += std::to_string(action + 1);
	this->player *= -1 ;
}

int ConnectFour::getEmptyColIndex(int action){
	if (this->allowedToPlay[action] == this->height){
		return -1;
	}
	return this->height - this->allowedToPlay[action] - 1;
}

bool ConnectFour::ended(){
	if(Position::alignment(this->board_rep.getCurrentPosition())){
		this->winner = this->player*-1;
		return true;
	} else if ((this->allowedToPlay == this->height).all()){
		this->winner = 0;
		return true;
	}
	
	return false;
}

int ConnectFour::getWinner(){
	return this->winner;
}


MatrixXf ConnectFour::getBoard(){
	return this->board;
}

float ConnectFour::getPlayer(){
	return this->player;
}

ArrayXf ConnectFour::getPossibleActions(){
	ArrayXf poss = ArrayXf::Zero(this->getActionSize());

	for(int i = 0; i < this->width; i++){
		poss[i] = this->board_rep.canPlay(i);
	}

	return poss;
}