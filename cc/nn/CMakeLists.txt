add_library(nnwrapper NNWrapper.cc NNWrapper.h utils.h)
target_link_libraries(nnwrapper "${TORCH_LIBRARIES}")
