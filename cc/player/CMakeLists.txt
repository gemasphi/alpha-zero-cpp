add_library(player Player.cc Player.h ${CONNECT_SOLVER_CPP})
target_include_directories(player PUBLIC ${connect4solver_SOURCE_DIR} ../nn ../games ../mcts)
target_link_libraries(player game nnwrapper mcts)
