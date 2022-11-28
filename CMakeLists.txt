cmake_minimum_required(VERSION 2.8.12)
project(EM_C++)

include_directories(include/)

set(CMAKE_CXX_FLAGS "-O3")

IF (NOT DEFINED SSE_SUPPORT)
    SET(SSE_SUPPORT 0)
ENDIF()
ADD_DEFINITIONS(-DSSE_SUPPORT=${SSE_SUPPORT})

IF (NOT DEFINED DEBUG)
    SET(DEBUG 0)
ENDIF()
ADD_DEFINITIONS(-DDEBUG=${DEBUG})

add_executable(RHEmc src/rhemc.cpp src/genotype.cpp src/storage.cpp)
add_executable(RHEmc_mem src/rhemc_mem_eff.cpp src/genotype.cpp src/storage.cpp)
add_executable(RHEmc_dom src/rhemc_dom.cpp src/genotype.cpp src/storage.cpp)
