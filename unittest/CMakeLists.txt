cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(cheddar_test LANGUAGES CXX)

option(ENABLE_EXTENSION "Enable extension sources" ON)

if(ENABLE_EXTENSION)
  add_compile_definitions(ENABLE_EXTENSION)
endif()
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

option(BUILD_TESTS OFF)
FetchContent_Declare(
  rmm
  GIT_REPOSITORY https://github.com/rapidsai/rmm
  GIT_TAG        branch-22.12
  GIT_SHALLOW
)
FetchContent_MakeAvailable(rmm)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
  jsoncpp
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)
FetchContent_Declare(
    cnpy
    GIT_REPOSITORY https://github.com/rogersce/cnpy
    GIT_TAG        master
)
FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.3.9
)
FetchContent_MakeAvailable(Eigen)
FetchContent_MakeAvailable(cnpy)
FetchContent_MakeAvailable(jsoncpp)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

find_package(CUDAToolkit 11.8 REQUIRED)
find_library(LIBTOMMATH tommath libtommath REQUIRED)

include_directories(/cheddar/include ${CUDAToolkit_INCLUDE_DIRS} 
                    ${LIBTOMMATH_INCLUDE_DIRS})
link_directories(/cheddar/lib)

add_executable(basic_test BasicTest.cpp)
target_link_libraries(basic_test
                      PRIVATE cheddar rmm gtest_main nlohmann_json::nlohmann_json)

if (ENABLE_EXTENSION)
  add_executable(boot_test Bootstrapping.cpp)
  add_executable(helr HELR.cpp)
  add_executable(resnet ResNet.cpp) 
  add_executable(sorting Sorting.cpp) 
  add_executable(sorting_prec Sorting_prec.cpp) 
  add_executable(resnet_acc ResNet_acc.cpp) 
  target_compile_definitions(helr PRIVATE PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
  target_compile_definitions(resnet PRIVATE PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
  target_compile_definitions(resnet_acc PRIVATE PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
  target_link_libraries(boot_test
                        PRIVATE cheddar rmm gtest_main nlohmann_json::nlohmann_json)
  target_link_libraries(helr 
                        PRIVATE cheddar rmm gtest_main nlohmann_json::nlohmann_json)
  target_include_directories(resnet PRIVATE ${cnpy_SOURCE_DIR})
  target_include_directories(resnet_acc PRIVATE ${cnpy_SOURCE_DIR})
  target_link_libraries(resnet
                        PRIVATE cheddar rmm gtest_main nlohmann_json::nlohmann_json cnpy eigen)
  target_link_libraries(resnet_acc
                        PRIVATE cheddar rmm gtest_main nlohmann_json::nlohmann_json cnpy eigen)
  target_link_libraries(sorting
                        PRIVATE cheddar rmm gtest_main nlohmann_json::nlohmann_json)
  target_link_libraries(sorting_prec
                        PRIVATE cheddar rmm gtest_main nlohmann_json::nlohmann_json)
endif()

configure_file(parameters/bootparam_30.json
               ${CMAKE_CURRENT_BINARY_DIR}/bootparam_30.json COPYONLY)
configure_file(parameters/bootparam_35.json
               ${CMAKE_CURRENT_BINARY_DIR}/bootparam_35.json COPYONLY)
configure_file(parameters/bootparam_40.json
               ${CMAKE_CURRENT_BINARY_DIR}/bootparam_40.json COPYONLY)
configure_file(parameters/bootparam_48.json
               ${CMAKE_CURRENT_BINARY_DIR}/bootparam_48.json COPYONLY)

configure_file(parameters/helrparam_30.json
               ${CMAKE_CURRENT_BINARY_DIR}/helrparam_30.json COPYONLY)   
configure_file(parameters/helrparam_35.json
               ${CMAKE_CURRENT_BINARY_DIR}/helrparam_35.json COPYONLY)   
configure_file(parameters/helrparam_40.json
               ${CMAKE_CURRENT_BINARY_DIR}/helrparam_40.json COPYONLY)   
configure_file(parameters/helrparam_48.json
               ${CMAKE_CURRENT_BINARY_DIR}/helrparam_48.json COPYONLY)   

configure_file(parameters/resnetparam_30.json
                ${CMAKE_CURRENT_BINARY_DIR}/resnetparam_30.json COPYONLY)   
configure_file(parameters/resnetparam_35.json
                ${CMAKE_CURRENT_BINARY_DIR}/resnetparam_35.json COPYONLY)   
configure_file(parameters/resnetparam_40.json
               ${CMAKE_CURRENT_BINARY_DIR}/resnetparam_40.json COPYONLY)   
configure_file(parameters/resnetparam_48.json
               ${CMAKE_CURRENT_BINARY_DIR}/resnetparam_48.json COPYONLY)   

configure_file(parameters/rnnparam_30.json
               ${CMAKE_CURRENT_BINARY_DIR}/rnnparam_30.json COPYONLY)   
configure_file(parameters/rnnparam_35.json
               ${CMAKE_CURRENT_BINARY_DIR}/rnnparam_35.json COPYONLY)   
configure_file(parameters/rnnparam_40.json
               ${CMAKE_CURRENT_BINARY_DIR}/rnnparam_40.json COPYONLY)   
configure_file(parameters/rnnparam_48.json
               ${CMAKE_CURRENT_BINARY_DIR}/rnnparam_48.json COPYONLY)   

configure_file(parameters/sortingparam_30.json
              ${CMAKE_CURRENT_BINARY_DIR}/sortingparam_30.json COPYONLY)   
configure_file(parameters/sortingparam_35.json
                  ${CMAKE_CURRENT_BINARY_DIR}/sortingparam_35.json COPYONLY)   
configure_file(parameters/sortingparam_40.json
               ${CMAKE_CURRENT_BINARY_DIR}/sortingparam_40.json COPYONLY)   
configure_file(parameters/sortingparam_48.json
               ${CMAKE_CURRENT_BINARY_DIR}/sortingparam_48.json COPYONLY)   

