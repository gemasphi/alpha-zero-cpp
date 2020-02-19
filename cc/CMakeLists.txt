cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(alphazero)

include(Dependencies.cmake)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

#-Wshadow
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
SET(CMAKE_CXX_STANDARD 17)

add_subdirectory(${CMAKE_SOURCE_DIR}/games)
add_subdirectory(${CMAKE_SOURCE_DIR}/nn)
add_subdirectory(${CMAKE_SOURCE_DIR}/mcts)
add_subdirectory(${CMAKE_SOURCE_DIR}/player)


add_executable(game_info game_info.cc)
add_dependencies(game_info json Eigen3 bayeselo)
target_include_directories(game_info PRIVATE ${CMAKE_SOURCE_DIR}/games ${JSON_INCLUDE_DIR})
target_link_libraries(game_info game)


set(BASE_LIBRARIES 
	OpenMP::OpenMP_CXX 
	-lstdc++fs
	-fopenmp
	game 
	mcts 
	player 
	nnwrapper
	connectfour
)

set(BASE_INCLUDE 
	${CMAKE_SOURCE_DIR}/games 
	${CMAKE_SOURCE_DIR}/nn 
	${CMAKE_SOURCE_DIR}/mcts
	${CMAKE_SOURCE_DIR}/player
	${JSON_INCLUDE_DIR}
	${CXXOPTS_INCLUDE_DIR}
	)


add_executable(selfplay selfplay.cc )
add_dependencies(selfplay json Eigen3 cxxopts)
target_include_directories(selfplay PUBLIC ${BASE_INCLUDE})
target_link_libraries(selfplay ${BASE_LIBRARIES})


add_executable(play_agaisnt play_agaisnt.cc)
add_dependencies(play_agaisnt json Eigen3 cxxopts)
target_include_directories(play_agaisnt PUBLIC ${BASE_INCLUDE})
target_link_libraries(play_agaisnt ${BASE_LIBRARIES})

add_executable(generate generate.cc)
add_dependencies(generate json Eigen3 cxxopts)
target_include_directories(generate PUBLIC ${BASE_INCLUDE})
target_link_libraries(generate ${BASE_LIBRARIES})