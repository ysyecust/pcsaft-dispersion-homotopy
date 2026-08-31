cmake_minimum_required(VERSION 3.16)
project(pcsaft_dispersion_homotopy CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(catalog STATIC
  src/benchmark_state_catalog.cpp
  src/difficult_state_catalog.cpp
  src/all_root_state_catalog.cpp)

# Smallest entry point: solve one state, print the classified root set.
add_executable(single_state_example src/single_state_example.cpp)
target_link_libraries(single_state_example PRIVATE catalog)

# Full method comparison on the frozen validation catalogue.
add_executable(external_baseline_benchmark src/external_baseline_benchmark.cpp)
target_link_libraries(external_baseline_benchmark PRIVATE catalog)

# Structural audit of the two homotopy embeddings.
add_executable(embedding_pole_audit src/embedding_pole_audit.cpp)
target_link_libraries(embedding_pole_audit PRIVATE catalog)

enable_testing()
foreach(t stationary_root_solver direct_root_methods reference_root_isolator
          external_baselines)
  add_executable(test_${t} tests/test_${t}.cpp)
  add_test(NAME ${t} COMMAND test_${t})
endforeach()
add_executable(test_complete_homotopy_curve tests/test_complete_homotopy_curve.cpp)
target_link_libraries(test_complete_homotopy_curve PRIVATE catalog)
add_test(NAME complete_homotopy_curve COMMAND test_complete_homotopy_curve)
