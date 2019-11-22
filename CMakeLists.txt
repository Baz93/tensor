project(tensor)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -pedantic -Wold-style-cast")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wconversion -Wshadow -Wunreachable-code -Wno-missing-braces")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=deprecated-declarations")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

add_subdirectory(googletest)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

set(SOURCE_FILES main.cpp src/tensor.h src/element_wise_operations.h src/tensor_common.h src/tensor_iterable.h src/tensor_subslice.h src/tensor_iterator.h tensor.h src/tensor_container.h src/undefs.h src/tensor_operations.h)
add_executable(Google_Tests_run ${SOURCE_FILES})
target_link_libraries(Google_Tests_run gtest gtest_main)
