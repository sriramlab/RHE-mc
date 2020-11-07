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

add_executable(RHEmc src/rhemc_cov_ldsc_her_def.cpp src/genotype.cpp src/storage.cpp)
add_executable(RHEmc_dom src/dom_jack_stream.cpp src/genotype.cpp src/storage.cpp)
