#include <Tictactoe.h>
#include <iostream>
#include "../eigen/Eigen/Dense"
#include "../eigen/Eigen/Core"
#include <stdexcept>
#include <sstream>
#include <memory>

using namespace Eigen;

TicTacToe::TicTacToe(int boardSize, float player){
	this->board = MatrixXf::Zero(boardSize, boardSize);
	this->boardSize = boardSize;
	this->player = 1;
}

std::unique_ptr<Game> TicTacToe::copy(){
	return std::make_unique<TicTacToe>(*this);
}

void TicTacToe::printBoard(){
	std::cout<< this->board << std::endl;
}

void TicTacToe::play(int action){
	int x = action/boardSize;
	int y = action%boardSize;

	if (this->board(x,y) != 0){
		std::ostringstream  error;
		error << "Invalid action: " << action << "\n" << this->board; 
		throw std::invalid_argument(error.str());
	}

	this->board(x,y) = this->player;
	this->player *= -1 ;
}

bool TicTacToe::ended(){
	for (int p : {-1, 1}){
		Matrix<bool,Dynamic,Dynamic> playerPositions = (this->board.array() == p).cast<bool>();

		if (this->findWin(playerPositions)){
			this->winner = p * this->player;
			return true;
		};
						

		if((this->board.array() == 0).count() == 0){
			this->winner = 0;
			return true;
		}								
	}

	return false;
}

int TicTacToe::getWinner(){
	return this->winner;
}

int TicTacToe::getCanonicalWinner(){
	return this->winner*this->player;
}
bool TicTacToe::findWin(Matrix<bool,Dynamic,Dynamic> playerPositions){
	for (int i = 0; i < this->boardSize - this->inRow + 1; i++){
		for (int j = 0; j < this->boardSize - this->inRow + 1; j++){
			if(this->isWin(playerPositions.block(i, j, this->inRow, this->inRow))){
				return true;
			}
		}
	}

	return false;
}

bool TicTacToe::isWin(Matrix<bool,Dynamic,Dynamic> smallBoard){
	Matrix<bool, Dynamic, Dynamic> vertical = smallBoard.rowwise().all();
	Matrix<bool, Dynamic, Dynamic> horizontal = smallBoard.colwise().all();
	bool ldiagonal = smallBoard.diagonal().all();
	bool rdiagonal = smallBoard.rowwise().reverse().diagonal().all();

	return vertical.any() or horizontal.any() or ldiagonal or rdiagonal;
}	

int TicTacToe::getActionSize(){
	return this->boardSize*this->boardSize;
}

int TicTacToe::getBoardSize(){
	return this->boardSize;
}

MatrixXf TicTacToe::getBoard(){
	return this->board;
}

float TicTacToe::getPlayer(){
	return this->player;
}

ArrayXf TicTacToe::getPossibleActions(){
	ArrayXf poss = ArrayXf::Zero(this->getActionSize());
	int bsize = (this->board).size();
	for(int i = 0; i < bsize; i++){
		if (this->board(i/boardSize, i%boardSize) == 0){
			poss(i) = 1;
		}
		else{
			poss(i) = 0;
		}
	}

	return poss;
}