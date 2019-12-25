cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(alphazero)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package(OpenMP REQUIRED)
find_package(Torch REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/games)
include_directories(${CMAKE_SOURCE_DIR}/games/connect4solver)

add_executable(selfplay selfplay.cc MCTS.cc NNWrapper.cc ${CMAKE_SOURCE_DIR}/games/Tictactoe.cc ${CMAKE_SOURCE_DIR}/games/ConnectFour.cc ${CMAKE_SOURCE_DIR}/games/Game.cc )
target_link_libraries(selfplay ${TORCH_LIBRARIES})
target_link_libraries(selfplay -lstdc++fs)
target_link_libraries(selfplay OpenMP::OpenMP_CXX)
set_property(TARGET selfplay PROPERTY CXX_STANDARD 17)



add_executable(play_agaisnt play_agaisnt.cc MCTS.cc Player.cc NNWrapper.cc ${CMAKE_SOURCE_DIR}/games/Tictactoe.cc ${CMAKE_SOURCE_DIR}/games/ConnectFour.cc ${CMAKE_SOURCE_DIR}/games/Game.cc ${CMAKE_SOURCE_DIR}/games/connect4solver/Solver.cpp)
target_link_libraries(play_agaisnt ${TORCH_LIBRARIES})
target_link_libraries(play_agaisnt -lstdc++fs)
target_link_libraries(play_agaisnt OpenMP::OpenMP_CXX)
set_property(TARGET play_agaisnt PROPERTY CXX_STANDARD 17)

add_executable(game_info ${CMAKE_SOURCE_DIR}/games/game_info.cc ${CMAKE_SOURCE_DIR}/games/Tictactoe.cc ${CMAKE_SOURCE_DIR}/games/ConnectFour.cc ${CMAKE_SOURCE_DIR}/games/Game.cc)
target_link_libraries(game_info -lstdc++fs)
set_property(TARGET game_info PROPERTY CXX_STANDARD 17)


