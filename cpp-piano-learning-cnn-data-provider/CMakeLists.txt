cmake_minimum_required(VERSION 2.8.12)
project(cpp_piano_learning_cnn_data_provider)

add_subdirectory(pybind11)
pybind11_add_module(cpp_piano_learning_cnn_data_provider src/main.cpp)

SET(cellar_dir "/usr/local/Cellar")

set(Boost_INCLUDE_DIR ${cellar_dir}/boost/1.65.1/include)
set(BOOST_LIBRARYDIR ${cellar_dir}/boost/1.65.1/lib)
set(Boost_LIBRARY_DIR ${cellar_dir}/boost/1.65.1/lib)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})
find_package(Boost COMPONENTS system serialization filesystem REQUIRED)

include_directories(src/lib-src/header-only)

ADD_LIBRARY(libsndfile SHARED IMPORTED)
SET_PROPERTY(TARGET libsndfile PROPERTY IMPORTED_LOCATION ${cellar_dir}/libsndfile/1.0.28/lib/libsndfile.dylib)
INCLUDE_DIRECTORIES(include ${cellar_dir}/libsndfile/1.0.28/include)

TARGET_LINK_LIBRARIES(cpp_piano_learning_cnn_data_provider PRIVATE libsndfile ${Boost_LIBRARIES})

