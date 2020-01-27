add_library(mcts MCTS.cc MCTS.h)
target_include_directories(mcts PRIVATE ../nn ../games)
target_link_libraries(mcts game nnwrapper)
