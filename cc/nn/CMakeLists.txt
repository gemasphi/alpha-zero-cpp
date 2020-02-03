
add_library(nnwrapper NNWrapper.cc )
target_include_directories(nnwrapper PUBLIC ../mcts)
target_link_libraries(nnwrapper gamestate "${TORCH_LIBRARIES}")
