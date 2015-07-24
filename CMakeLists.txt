cmake_minimum_required(VERSION 2.8.12)
project(distance-transform)


add_executable(test1 test.c dist.c dist.h)

set(tgt dist-avx2)
add_library(${tgt} STATIC dist.avx2.c )
if(WIN32)
# don't really need /arhc:avx2.  Using will enable avx2 code generation in debug builds
else()
target_compile_options(${tgt} PRIVATE -mavx2)
endif()

add_executable(test2 test.c dist.h)
target_link_libraries(test2 dist-avx2)
