cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(alphazero)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(${CMAKE_SOURCE_DIR}/games)
add_executable(mcts MCTS.cc NNWrapper.cc ${CMAKE_SOURCE_DIR}/games/Tictactoe.cc )

target_link_libraries(mcts "${TORCH_LIBRARIES}")
set_property(TARGET mcts PROPERTY CXX_STANDARD 14)


