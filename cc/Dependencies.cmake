include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)
include(${CMAKE_ROOT}/Modules/FetchContent.cmake)

set( Eigen3_VERSION "3.2.9" )

ExternalProject_Add( Eigen3
  URL "http://bitbucket.org/eigen/eigen/get/${Eigen3_VERSION}.tar.gz"
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND
    ${CMAKE_COMMAND} -E copy_directory
      ${CMAKE_BINARY_DIR}/Eigen3-prefix/src/Eigen3/Eigen
      ${INSTALL_DEPENDENCIES_DIR}/include/Eigen3/Eigen &&
    ${CMAKE_COMMAND} -E copy_directory
      ${CMAKE_BINARY_DIR}/Eigen3-prefix/src/Eigen3/unsupported
      ${INSTALL_DEPENDENCIES_DIR}/include/Eigen3/unsupported
)

set(EIGEN3_INCLUDE_DIR ${INSTALL_DEPENDENCIES_DIR}/include/Eigen3 )

ExternalProject_Add(json
  URL "https://github.com/nlohmann/json/releases/download/v3.7.3/include.zip"
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND
   ${CMAKE_COMMAND} -E copy
      ${CMAKE_BINARY_DIR}/json-prefix/src/json/single_include/nlohmann/json.hpp
      ${INSTALL_DEPENDENCIES_DIR}/include/nlohmann/json.hpp
 )


set(JSON_INCLUDE_DIR  ${INSTALL_DEPENDENCIES_DIR}/include/nlohmann/)

list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR}/libtorch)
find_package(Torch)
if(NOT Torch_FOUND)
  file(DOWNLOAD 
    https://download.pytorch.org/libtorch/nightly/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip 
    ${CMAKE_CURRENT_BINARY_DIR}/libtorch
    TIMEOUT 60  # seconds
    TLS_VERIFY ON
  )

  execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xf libtorch 
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  find_package(Torch REQUIRED)

endif()
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package(OpenMP REQUIRED)


FetchContent_Declare(
  connect4solver
  GIT_REPOSITORY https://github.com/gemasphi/connect4
)

FetchContent_GetProperties(connect4solver)
if(NOT connect4solver_POPULATED)
  FetchContent_Populate(connect4solver)
endif()

FILE(GLOB CONNECT_SOLVER_CPP  "${connect4solver_SOURCE_DIR}/*.cpp")
message(CONNECT_SOLVER_CPP)