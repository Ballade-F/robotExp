cmake_minimum_required(VERSION 3.5)
project(robot)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_STANDARD 17)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
# find_package(matplotlib_cpp REQUIRED)
# find_package(Python3 COMPONENTS Development NumPy REQUIRED)
find_package(message REQUIRED)
set(Torch_DIR ${CMAKE_SOURCE_DIR}/thirdparty/libtorch/share/cmake/Torch) 
find_package(Torch REQUIRED)

include_directories(
  /usr/include/eigen3
  /usr/include/python3.8
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/include/robot
  ${CMAKE_SOURCE_DIR}/thirdparty/include
  ${CMAKE_SOURCE_DIR}/thirdparty/include/json
  ${CMAKE_SOURCE_DIR}/thirdparty/include/osqp
  ${CMAKE_SOURCE_DIR}/thirdparty/include/qdldl
  ${CMAKE_SOURCE_DIR}/thirdparty/include/libusb-1.0
)

file(GLOB ROBOT_SOURCEFILE  
  ${CMAKE_SOURCE_DIR}/src/*.cpp)
add_library(ROBOT_LIB STATIC ${ROBOT_SOURCEFILE})
target_link_directories(ROBOT_LIB PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty/lib)
target_link_libraries(ROBOT_LIB jsoncpp osqp qdldl tinyxml2 ${TORCH_LIBRARIES})

add_executable(sim_node src/sim_map_node.cc)
target_link_libraries(sim_node ROBOT_LIB)
ament_target_dependencies(sim_node rclcpp std_msgs message)

add_executable(robot_node src/robot_node.cc)
target_link_directories(robot_node PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty/lib)
target_link_libraries(robot_node jsoncpp osqp qdldl tinyxml2 ROBOT_LIB ${TORCH_LIBRARIES})
ament_target_dependencies(robot_node rclcpp std_msgs message)

install(TARGETS 
  sim_node
  robot_node
  DESTINATION lib/${PROJECT_NAME})

install(
  DIRECTORY 
  ${CMAKE_SOURCE_DIR}/thirdparty/include/json
  ${CMAKE_SOURCE_DIR}/thirdparty/include/osqp
  ${CMAKE_SOURCE_DIR}/thirdparty/include/qdldl
  ${CMAKE_SOURCE_DIR}/thirdparty/include/libusb-1.0
  DESTINATION 
  include
)
install(FILES ${CMAKE_SOURCE_DIR}/thirdparty/include/tinyxml2.h DESTINATION include)
FILE (GLOB INSTALL_LIB ${CMAKE_SOURCE_DIR}/thirdparty/lib/*.*)
install(
  FILES 
  ${INSTALL_LIB}
  DESTINATION 
  lib
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
