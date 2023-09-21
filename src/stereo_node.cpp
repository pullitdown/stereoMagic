#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Image.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp> // 替换 #include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp> // 替换 #include <opencv/highgui.h>
#include <opencv2/calib3d/calib3d.hpp> // 替换 #include "opencv2/calib3d.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
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

using namespace message_filters;
using namespace sensor_msgs;
using namespace cv;
using namespace cv::ximgproc;
using namespace std;



bool enableVisualization = 0 ;
// uncomment to visualize results
static const std::string OPENCV_WINDOW = "Image window";


//publishers for image and pointcloud
image_transport::Publisher pub;
ros::Publisher pcpub;

//parameters for stereo matching and filtering
double vis_mult = 5.0;
int BLockSizr;
int wsize = 13;
int max_disp = 16 * 10;
double lambda = 10000.0;
double sigma = 1.0;


//Some object instatiation that can be done only once
Mat left_for_matcher,right_for_matcher;
Mat left_disp, right_disp;
Mat filtered_disp;
Rect ROI ;
Ptr<DisparityWLSFilter> wls_filter;
Mat filtered_disp_vis;
int sgbm_type;
int p1_num,p2_num,prefilter_num;
int Disp12MaxDiff,MinDisparity,SpeckleWindowSize,SpeckleRange,PreFilterCap,UniquenessRatio;




void compute_stereo(Mat& imL, Mat& imR)
{
  Mat Limage(imL);
  //confidence map
  Mat conf_map = Mat(imL.rows,imL.cols,CV_8U);
  conf_map = Scalar(255);

  // downsample images to speed up results
  max_disp/=2;
  if(max_disp%16 != 0) max_disp += 16-(max_disp%16);
  resize(imL, left_for_matcher,Size(),0.5,0.5);
  resize(imR, right_for_matcher,Size(),0.5,0.5);

  //compute disparity
  Ptr<StereoSGBM> left_matcher  = StereoSGBM::create(0,max_disp,wsize);
  left_matcher->setBlockSize(BlockSize)
  left_matcher->setP1(p1_num*wsize*wsize);
  left_matcher->setP2(p2_num*wsize*wsize);
  left_matcher->setDisp12MaxDiff(Disp12MaxDiff);
  left_matcher->setMinDisparity(MinDisparity);
  left_matcher->setSpeckleWindowSize(SpeckleWindowSize);
  left_matcher->setSpeckleRange(SpeckleRange);
  left_matcher->setPreFilterCap(PreFilterCap);
  left_matcher->setUniquenessRatio(UniquenessRatio);
  left_matcher->setMode(static_cast<decltype(StereoSGBM::MODE_SGBM_3WAY)>(sgbm_type) );
  wls_filter = createDisparityWLSFilter(left_matcher);
  Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);

  left_matcher-> compute(left_for_matcher, right_for_matcher,left_disp);
  right_matcher->compute(right_for_matcher,left_for_matcher, right_disp);

  //filter
  wls_filter->setLambda(lambda);
  wls_filter->setSigmaColor(sigma);
  wls_filter->filter(left_disp,imL,filtered_disp,right_disp);
  conf_map = wls_filter->getConfidenceMap();
  ROI = wls_filter->getROI();

  //visualization
  getDisparityVis(filtered_disp,filtered_disp_vis,vis_mult);


  //PointCloud Generation======================================================

  // Q matrix (guess until we can do the correct calib process)
  // double w = imR.cols;
  // double  h = imR.rows;
  // double f = 843.947693;
  // double cx = 508.062911;
  // double cx1 = 526.242457;
  // double cy = 385.070250;
  // double Tx = -120.00;
  // double w = imR.cols;
  // double  h = imR.rows;
  // double f = 382.34175236928627;
  // double cx = 321.7242056112083;
  // double cx1 = 322.3064389150983;
  // double cy = 236.24691075834372;
  // double Tx = -19.29692482;
  // Mat Q = Mat(4,4, CV_64F, double(0));
  // Q.at<double>(0,0) = 1.0;
  // Q.at<double>(0,3) = -cx;
  // Q.at<double>(1,1) = 1.0;
  // Q.at<double>(1,3) = -cy;
  // Q.at<double>(2,3) = f;
  // Q.at<double>(3,2) = -1.0/ Tx;
  // Q.at<double>(3,3) = ( cx - cx1)/ Tx;

  double Tx=-19.42830883756359;
  cv::Mat Q = (cv::Mat_<double>(4,4) << -19.42830883756359, 0, 0, 6251.133074507617,
                                     0, -19.42830883756359, 0, 4592.437057166097,
                                     0, 0, 0, -7472.829104043432,
                                     0, 0, -1, 11.40046179682074/384.6361083984375);
  Q=Q/Tx;
  std::cout<<Q<<std::endl;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud(new  pcl::PointCloud<pcl::PointXYZRGB>());
  Mat xyz;
  reprojectImageTo3D(filtered_disp, xyz, Q, true);
  pointcloud->width = static_cast<uint32_t>(filtered_disp.cols);
  pointcloud->height = static_cast<uint32_t>(filtered_disp.rows);
  pointcloud->is_dense = true;
  pcl::PointXYZRGB point;
  for (int i = 0; i < filtered_disp.rows; ++i)
  {
    uchar* rgb_ptr = Limage.ptr<uchar>(i);
    uchar* filtered_disp_ptr = filtered_disp.ptr<uchar>(i);
    double* xyz_ptr = xyz.ptr<double>(i);

    for (int j = 0; j < filtered_disp.cols; ++j)
    {

      uchar d = filtered_disp_ptr[j];
      //if (d == 0) continue;
      Point3f p = xyz.at<Point3f>(i, j);

      double radius = sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
      if(radius < 20*100)
      {
        point.z = p.z/100.0;
        point.x = p.x/100.0;
        point.y = p.y/100.0;
        point.b = rgb_ptr[ j];
        point.g = rgb_ptr[ j];
        point.r = rgb_ptr[ j];
        pointcloud->points.push_back(point);
      }

      else
      {
        point.z = 0.0;
        point.x = 0.0;
        point.y = 0.0;

        point.b = rgb_ptr[3 * j];
        point.g = rgb_ptr[3 * j];
        point.r = rgb_ptr[3 * j];
        pointcloud->points.push_back(point);
      }
    }
  }


  // voxel grid filter
  // pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new  pcl::PointCloud<pcl::PointXYZRGB>());
  // pcl::VoxelGrid<pcl::PointXYZRGB> sor;
  // sor.setInputCloud (pointcloud);
  // sor.setLeafSize (0.01, 0.01, 0.01);
  // sor.filter (*cloud_filtered);


  // //outliner removal filter
  // pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered2(new  pcl::PointCloud<pcl::PointXYZRGB>());
  // pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor1;
  // sor1.setInputCloud (cloud_filtered);
  // sor1.setMeanK (100);
  // sor1.setStddevMulThresh (0.001);
  // sor1.filter (*cloud_filtered2);
  // if(enableVisualization)
  // {
  //   pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");
  //   cv::imshow(OPENCV_WINDOW, filtered_disp_vis);
  //   viewer.showCloud(cloud_filtered2);
  //   cv::waitKey(3);
  // }

   // Convert to ROS data type
   sensor_msgs::PointCloud2 pointcloud_msg;
   pcl:: toROSMsg(*pointcloud,pointcloud_msg);
   pointcloud_msg.header.frame_id = "cam_pos";

   // Publishes pointcloud message
   pcpub.publish(pointcloud_msg);


}





void callback(const ImageConstPtr& left, const ImageConstPtr& right) {
  // conversion to rosmsgs::Image to cv::Mat using cv_bridge

  cv_bridge::CvImagePtr cv_left;
  try
    {
      cv_left = cv_bridge::toCvCopy(left);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

  cv_bridge::CvImagePtr cv_right;
  try
      {
        cv_right = cv_bridge::toCvCopy(right);
      }
      catch (cv_bridge::Exception& e)
      {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
      }


   compute_stereo(cv_left->image,cv_right->image);
   sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "mono8",filtered_disp_vis).toImageMsg();
   pub.publish(msg);

}

int main(int argc, char **argv) {

  ros::init(argc, argv, "stereo_node");
	ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);

  //pointcloud publisher


  std::string left_topic, right_topic,pc_topic,depth_topic;
  nh.param<std::string>("left_topic", left_topic, "/infra1/image_rect_raw");
  nh.param<std::string>("right_topic", right_topic, "/infra2/image_rect_raw");
  nh.param<std::string>("pc_topic", pc_topic, "/camera/depth/points");
  nh.param<std::string>("depth_topic", depth_topic, "/camera/depth/image");
  nh.param<int>("p1_num",p1_num,24);
  nh.param<int>("p2_num",p2_num,96);
  nh.param<int>("sgbm_type",sgbm_type,0) ;
  nh.param<int>("prefilter_num",prefilter_num,63);
  nh.param<int>("Disp12MaxDiff",Disp12MaxDiff,20);
  nh.param<int>("MinDisparity",MinDisparity,5);
  nh.param<int>("SpeckleWindowSize",SpeckleWindowSize,100);
  nh.param<int>("SpeckleRange",SpeckleRange,1);
  nh.param<int>("PreFilterCap",PreFilterCap,63);
  nh.param<int>("UniquenessRatio",UniquenessRatio,10);
  
  
  pcpub = nh.advertise<sensor_msgs::PointCloud2> (pc_topic, 1);
  // depth image publisher
  pub = it.advertise(depth_topic, 1);
  //left and right rectified images subscriber
	message_filters::Subscriber<Image> left_sub(nh, left_topic, 1);
	message_filters::Subscriber<Image> right_sub(nh, right_topic, 1);

  //time syncronizer to publish 2 images in the same callback function
	TimeSynchronizer<Image, Image> sync(left_sub, right_sub, 1);

  //call calback each time a new message arrives
  sync.registerCallback(boost::bind(&callback, _1, _2));

	ros::spin();
	return 0;
}
