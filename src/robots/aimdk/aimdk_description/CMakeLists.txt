cmake_minimum_required(VERSION 3.10)
project(aimdk_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  urdf
  meshes
  DESTINATION share/${PROJECT_NAME}/
)
install(FILES LICENSE manifest.yaml DESTINATION share/${PROJECT_NAME}/)

ament_package()
