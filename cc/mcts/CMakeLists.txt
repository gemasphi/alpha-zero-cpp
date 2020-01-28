add_library(gamestate GameState.cc GameState.h)
target_include_directories(gamestate PUBLIC  ../nn ../games)
target_link_libraries(gamestate game nnutils)


add_library(mcts MCTS.cc MCTS.h GameState.h)
target_include_directories(mcts PUBLIC ../nn ../games)
target_link_libraries(mcts game nnwrapper gamestate)
