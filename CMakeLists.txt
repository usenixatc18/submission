project(self_adaptive_simulator)
cmake_minimum_required(VERSION 3.0)

find_package(Boost REQUIRED COMPONENTS system program_options filesystem) 
find_package(nlohmann_json REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_CXX_FLAGS 
    "-std=c++17 -Wall -Wextra -Werror -Wsuggest-override ${CMAKE_CXX_FLAGS}")

set(CMAKE_CXX_FLAGS_DEBUG 
    "-fsanitize=address ${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_RELEASE 
    "-O2 -Wno-unused-parameter ${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE ${CMAKE_EXE_LINKER_FLAGS_RELEASE})
    

add_subdirectory(src)
