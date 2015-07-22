cmake_minimum_required(VERSION 2.8)
project(distance-transform)
add_executable(test1 test.c dist.c dist.h)
add_executable(test2 test.c dist.avx2.c dist.h)
