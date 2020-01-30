add_library(main main.cc)
target_link_libraries(main Catch)

add_executable(mcts_test mcts)
target_include_directories(mcts_test PUBLIC ../mcts)
target_link_libraries(mcts_test main mcts gamestate Catch)