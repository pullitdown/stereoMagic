#ifndef STEREO_NODELET_H_
#define STEREO_NODELET_H_

#include <ros/ros.h>
#include <nodelet/nodelet.h> // 基类Nodelet所在的头文件。
#include <std_msgs/Float64.h>
#include <ros/callback_queue.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Image.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/ximgproc/disparity_filter.hpp"
#include <iostream>
#include <string>
#include <math.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <image_transport/subscriber_filter.h>
namespace magicStereo
{
    class magic_stereo_nodelet_test : public nodelet::Nodelet // 任何nodelet plugin都要继承Nodelet类。
    {
    public:
        magic_stereo_nodelet_test()
        {
        }
         void compute_stereo(cv::Mat &imL, cv::Mat &imR);
         void callback(const sensor_msgs::ImageConstPtr &left, const sensor_msgs::ImageConstPtr &right);
         void singleimagecb(const sensor_msgs::ImageConstPtr &left);
         void depthCallback(const sensor_msgs::ImageConstPtr &depth);
         void depthLoop(ros::NodeHandle &nb);
    private:
        virtual void onInit(); // 此函数声明部分为固定格式，在nodelet加载此plugin会自动执行此函数。
        bool enableVisualization = 0;
        // uncomment to visualize results
        std::string OPENCV_WINDOW = "Image window";
        std::unique_ptr<boost::thread> depth_thread_ ;
        // publishers for image and pointcloud
        image_transport::Publisher pub;
        ros::Publisher pcpub;
        std::string left_image_topic,right_image_topic;

        // parameters for stereo matching and filtering
        double vis_mult = 5.0;
        int wsize = 13;
        int max_disp = 16 * 10;
        double lambda = 10000.0;
        double sigma = 1.0;

        // Some object instatiation that can be done only once
        cv::Mat left_for_matcher, right_for_matcher;
        cv::Mat left_disp, right_disp;
        cv::Mat filtered_disp;
        cv::Rect ROI;
        cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter;
        cv::Mat filtered_disp_vis;
    };

}

#endif