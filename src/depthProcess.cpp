#include <ros/ros.h>
#include <stdio.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <yaml-cpp/yaml.h>


void depthCallback(const sensor_msgs::ImageConstPtr& msg)
{
       ROS_INFO_STREAM("normal subscribe address is"<<msg<<"  data address is "<<&(msg->data));
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "matrix_transform");
  // YAML::Node config = YAML::LoadFile("/media/world/lively_ws/src/lively_slam/param/euroc_stereo.yaml");
  //ROS_INFO_STREAM(config["cameras"][0]["T_B_C"]["date"].Type());
  
  //Convert OpenCV image to ROS message
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber left_sub_=it.subscribe( "/camera/depth/image", 1,boost::bind<void ,const sensor_msgs::ImageConstPtr&>(depthCallback,_1));

  //回调函数 
  ros::spin();
  return 0;
}