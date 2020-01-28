add_library(nnutils NNUtils.h)
target_link_libraries(nnutils PUBLIC "${TORCH_LIBRARIES}")
set_target_properties(nnutils PROPERTIES LINKER_LANGUAGE CXX)

add_library(nnwrapper NNWrapper.cc NNWrapper.h utils.h)
target_include_directories(nnwrapper PUBLIC ../mcts)
target_link_libraries(nnwrapper gamestate "${TORCH_LIBRARIES}")
