cmake_minimum_required(VERSION 3.22)
project(ruka_gz)

find_package(ament_cmake REQUIRED)


if(CMAKE_CXX_COMPILER_ID MATCHES "(GNU|Clang)")
  add_compile_options(-Wall -Wextra)
endif()

set(PACKAGE_INCLUDE_DEPENDS
   hardware_interface
  controller_interface
 # semantic_components
  sensor_msgs
  pluginlib
  rclcpp
  rclcpp_lifecycle
)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)

foreach(Dependency IN ITEMS ${PACKAGE_INCLUDE_DEPENDS})
  find_package(${Dependency} REQUIRED)
endforeach()

add_library(
  ruka_gz
  SHARED
  hardware/ruka_hardware.cpp
)

target_compile_features(ruka_gz PUBLIC cxx_std_17)
target_include_directories(ruka_gz PUBLIC
$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/hardware/include>

$<INSTALL_INTERFACE:include/ruka_gz>
)
ament_target_dependencies(
  ruka_gz PUBLIC
  ${PACKAGE_INCLUDE_DEPENDS}
)

target_compile_definitions(${PROJECT_NAME} PRIVATE "RUKA_BUILDING_DLL")

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(DIRECTORY world DESTINATION share/${PROJECT_NAME})


install(
  DIRECTORY hardware/include/
  DESTINATION include/ruka_gz
)

install(TARGETS ruka_gz
  EXPORT export_ruka_gz
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
endif()

## EXPORTS
ament_export_targets(export_ruka_gz HAS_LIBRARY_TARGET)
ament_export_dependencies(${PACKAGE_INCLUDE_DEPENDS})




ament_package()
