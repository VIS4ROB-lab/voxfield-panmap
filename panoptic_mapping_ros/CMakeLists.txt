cmake_minimum_required(VERSION 3.0.2)
project(panoptic_mapping_ros)

set(CMAKE_CXX_STANDARD 17)
find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)
catkin_package()

#############
# Libraries #
#############

cs_add_library(${PROJECT_NAME}
        src/panoptic_mapper.cpp
        src/input/input_synchronizer.cpp
        src/visualization/submap_visualizer.cpp
        src/visualization/single_tsdf_visualizer.cpp
        src/visualization/planning_visualizer.cpp
        src/visualization/tracking_visualizer.cpp
        src/conversions/conversions.cpp
        src/transformer/transformer.cpp
        )

###############
# Executables #
###############

cs_add_executable(panoptic_mapper_node
        app/panoptic_mapper_node.cpp
        )
target_link_libraries(panoptic_mapper_node ${PROJECT_NAME})

###########
# Export #
###########

cs_install()
cs_export()

