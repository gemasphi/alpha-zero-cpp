add_library(game Game.cc Game.h Tictactoe.cc Tictactoe.h)

add_library(tictactoe Tictactoe.cc Game.h Tictactoe.h)

add_library(connectfour ConnectFour.cc Game.h ConnectFour.h ${CONNECT_SOLVER_CPP})
target_include_directories(connectfour PUBLIC ${connect4solver_SOURCE_DIR})
