cmake_minimum_required(VERSION 3.12)

project(MisGreedySolver)

find_package(pybind11 REQUIRED)


pybind11_add_module(mis_greedy_cpp solver.cpp bindings.cpp)
install(TARGETS mis_greedy_cpp
        LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/)