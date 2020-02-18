
add_library(tictactoe Tictactoe.cc Game.h Tictactoe.h)

add_library(connectfour ConnectFour.cc Game.h ConnectFour.h ${CONNECT_SOLVER_CPP})
target_include_directories(connectfour PUBLIC ${connect4solver_SOURCE_DIR})

add_library(game Game.cc Game.h Tictactoe.cc Tictactoe.h )
target_include_directories(game PUBLIC ${connect4solver_SOURCE_DIR})
target_link_libraries(game connectfour)

pybind11_add_module(game_py pybind.cc)
target_link_libraries(game_py PUBLIC tictactoe)
set_target_properties(tictactoe PROPERTIES POSITION_INDEPENDENT_CODE TRUE)