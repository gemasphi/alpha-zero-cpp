add_library(gamestate GameState.cc)
target_include_directories(gamestate PUBLIC ../games)
target_link_libraries(gamestate game OpenMP::OpenMP_CXX 
	-lstdc++fs)


add_library(mcts MCTS.cc)
target_include_directories(mcts PUBLIC ../nn ../games ${CXXOPTS_INCLUDE_DIR})
target_link_libraries(mcts game nnwrapper gamestate OpenMP::OpenMP_CXX 
	-lstdc++fs)
