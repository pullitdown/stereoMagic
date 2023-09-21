cmake_minimum_required(VERSION 2.8.3)
project(stereo)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")



find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  message_filters
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf
  pcl_ros
  pcl_conversions
  dynamic_reconfigure
)



generate_dynamic_reconfigure_options(
  cfg/dynamic.cfg
)

catkin_package()





###########
## Build ##
###########



find_package(PCL  REQUIRED)
find_package(OpenCV  REQUIRED)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  include/stereoMagic
  ${CATKIN_DEVEL_PREFIX}/include

)

## Declare a C++ executable
add_executable(stereo_node_ src/stereo_node_.cpp)
# add_executable(calib_camera src/calib_camera.cpp)
# add_executable(stereo_tf_broadcaster src/stereo_tf_broadcaster.cpp)
# add_executable(draw_boxes src/draw_boxes.cpp)



# add_executable(normal_depth_callback src/depthProcess)
# target_link_libraries(normal_depth_callback ${catkin_LIBRARIES})


# foreach(item ${catkin_LIBRARIES})
#       message(${item})
# endforeach()
# foreach(item ${OpenCV_LIBRARIES})
#       message(${item})
# endforeach()
# foreach(item ${PCL_LIBRARIES})
#       message(${item})
# endforeach()

# ## Specify libraries to link a library or executable target against
# target_link_libraries(stereo_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})
# target_link_libraries(calib_camera ${catkin_LIBRARIES} ${OpenCV_LIBS})
# target_link_libraries(stereo_tf_broadcaster ${catkin_LIBRARIES} ${OpenCV_LIBS})
# target_link_libraries(draw_boxes ${catkin_LIBRARIES} ${OpenCV_LIBS})

foreach(item ${catkin_LIBRARIES})
      message(${item} " ")
endforeach()

# foreach(item ${PCL_LIBRARIES})
#       message(${item})
# endforeach()

## Specify libraries to link a library or executable target against
target_link_libraries(stereo_node_ ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS}
)

# target_link_libraries(calib_camera ${catkin_LIBRARIES} ${OpenCV_LIBS})
# target_link_libraries(stereo_tf_broadcaster ${catkin_LIBRARIES} ${OpenCV_LIBS})
# target_link_libraries(draw_boxes ${catkin_LIBRARIES} ${OpenCV_LIBS})


# #nodelet library 
# add_library(stereo_nodelet src/stereo_nodelet.cpp)
# add_dependencies(stereo_nodelet ${catkin_EXPORTED_TARGETS})
# target_link_libraries(stereo_nodelet ${catkin_LIBRARIES} ${PCL_LIBRARIES}  ${OpenCV_LIBS})

# add_library(stereo_nodelet_test src/stereo_nodelet_test.cpp)
# target_link_libraries(stereo_nodelet_test ${catkin_LIBRARIES} ${PCL_LIBRARIES}  ${OpenCV_LIBS})