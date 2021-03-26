cmake_minimum_required(VERSION 2.8.6)
project(window)

find_package(pybind11 REQUIRED)

add_compile_options(-DHAVE_CSTDDEF)
include_directories(${DART_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIR})
find_package(DART REQUIRED COMPONENTS)
find_package(pybind11 REQUIRED)

file(GLOB srcs "*.h" "*.cpp")

add_executable(window ${srcs})
target_link_libraries(window
	${DART_LIBRARIES} 
	${PYTHON_LIBRARIES}
)
