cmake_minimum_required(VERSION 3.5)
project(message)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../ros2support.cmake)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
FIND_PACKAGES(
  ament_cmake
  rclcpp
  geometry_msgs
  rosidl_default_generators
  tf2_geometry_msgs
)

file(GLOB ROBOT_MESSAGES_SOURCE "msg/*.msg" "srv/*.srv" "action/*.action")
GEN_MESSAGE(MESSAGES ${ROBOT_MESSAGES_SOURCE} DEPENDENCIES geometry_msgs )


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
