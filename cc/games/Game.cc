#include <iostream>
#include <iostream>
#include "Game.h"
#include "Tictactoe.h"
//#include "ConnectFour.h"
#include <memory>


std::unique_ptr<Game> Game::create(std::string type){
 	if (type == "TICTACTOE") 
        return std::make_unique<TicTacToe>(); 
    //todo
   /* else if (type == "CONNECTFOUR") 
        return std::make_unique<ConnectFour>();*/ 
 }