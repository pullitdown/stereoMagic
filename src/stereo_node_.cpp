#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Image.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>       // 替换 #include <opencv/cv.h>
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
#include <image_geometry/stereo_camera_model.h>
#include <yaml-cpp/yaml.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <dynamic_reconfigure/server.h>
#include <stereo/dynamicConfig.h>
using namespace message_filters;
using namespace sensor_msgs;
using namespace cv;
using namespace cv::ximgproc;
using namespace std;

class stereo_node_
{

public:
  stereo_node_(ros::NodeHandle &nh) : it(nh), pnh("~")
  {

    std::string left_topic, right_topic, pc_topic, depth_topic;
    pnh.param<std::string>("left_topic", left_topic, "/infra1/image_rect_raw");

    pnh.param<std::string>("right_topic", right_topic, "/infra2/image_rect_raw");
    pnh.param<std::string>("pc_topic", pc_topic, "/camera/depth/points");
    pnh.param<std::string>("depth_topic", depth_topic, "/camera/depth/image");
    pnh.param<int>("p1_num", p1_num, 24);
    pnh.param<int>("p2_num", p2_num, 96);
    pnh.param<int>("sgbm_type", sgbm_type, 0);
    pnh.param<int>("prefilter_num", prefilter_num, 63);
    pnh.param<int>("Disp12MaxDiff", Disp12MaxDiff, 20);
    pnh.param<int>("MinDisparity", MinDisparity, 5);
    pnh.param<int>("SpeckleWindowSize", SpeckleWindowSize, 100);
    pnh.param<int>("SpeckleRange", SpeckleRange, 1);
    pnh.param<int>("PreFilterCap", PreFilterCap, 63);
    pnh.param<int>("UniquenessRatio", UniquenessRatio, 10);
    pnh.param<int>("BlockSize", BlockSize, 13);
    ROS_INFO("topic: %s %s", right_topic.c_str(), left_topic.c_str());
    std::cout << right_topic << left_topic << std::endl;
    dynamic_reconfigure::Server<stereo::dynamicConfig>::CallbackType f = boost::bind(&stereo_node_::dynamic_reconfigure_callback, this, _1, _2);
    this->server.setCallback(f);

    pcpub = nh.advertise<sensor_msgs::PointCloud2>(pc_topic, 1);
    // depth image publisher
    pub = it.advertise(depth_topic, 1);
    // left and right rectified images subscriber
    left_sub.reset(new message_filters::Subscriber<Image>(nh, left_topic, 1));
    right_sub.reset(new message_filters::Subscriber<Image>(nh, right_topic, 1));
    std::string camera_param_path;
    pnh.param<std::string>("camera_param_path", camera_param_path, "/home/sunteng/catkin_ws/src/rpg_svo_pro_open/svo_ros/param/calib/realsense_stereo_best.yaml");

    std::vector<sensor_msgs::CameraInfo> camera_infos_;
    // time syncronizer to publish 2 images in the same callback function
    readSVOCameraParamFile(camera_param_path, camera_infos_);

    model_.fromCameraInfo(camera_infos_[0], camera_infos_[1]);
    sync.reset(new TimeSynchronizer<Image, Image>(*left_sub, *right_sub, 1));
    // call calback each time a new message arrives
    sync->registerCallback(boost::bind(&stereo_node_::callback, this, _1, _2));
  }

  void readSVOCameraParamFile(const std::string &path, std::vector<sensor_msgs::CameraInfo> &camera)
  {
    cv::Mat P0, P1, R0, R1, K0, K1, D0, D1;
    sensor_msgs::CameraInfo pcam1, pcam0;
    cv::Mat pose_w_cam0, pose_w_cam1;
    cv::Mat k_seq_0, k_seq_1;
    std::string model_dist;
    try
    {
      YAML::Node config = YAML::LoadFile(path);

      std::vector<double> T_B_C0_data = config["cameras"][0]["T_B_C"]["data"].as<std::vector<double>>();
      pose_w_cam0 = (cv::Mat_<double>(3, 4) << T_B_C0_data[0], T_B_C0_data[1], T_B_C0_data[2], T_B_C0_data[3],
                     T_B_C0_data[4], T_B_C0_data[5], T_B_C0_data[6], T_B_C0_data[7],
                     T_B_C0_data[8], T_B_C0_data[9], T_B_C0_data[10], T_B_C0_data[11]);

      std::vector<double> T_B_C1_data = config["cameras"][1]["T_B_C"]["data"].as<std::vector<double>>();
      pose_w_cam1 = (cv::Mat_<double>(3, 4) << T_B_C1_data[0], T_B_C1_data[1], T_B_C1_data[2], T_B_C1_data[3],
                     T_B_C1_data[4], T_B_C1_data[5], T_B_C1_data[6], T_B_C1_data[7],
                     T_B_C1_data[8], T_B_C1_data[9], T_B_C1_data[10], T_B_C1_data[11]);

      model_dist = config["cameras"][1]["camera"]["distortion"]["type"].as<std::string>();

      std::vector<double> intrinsics0 = config["cameras"][0]["camera"]["intrinsics"]["data"].as<std::vector<double>>();
      k_seq_0 = (cv::Mat_<double>(1, 4) << intrinsics0[0], intrinsics0[1], intrinsics0[2], intrinsics0[3]);

      // 读取相机0的畸变参数
      std::vector<double> distortion0 = config["cameras"][0]["camera"]["distortion"]["parameters"]["data"].as<std::vector<double>>();
      D0 = (cv::Mat_<double>(1, 4) << distortion0[0], distortion0[1], distortion0[2], distortion0[3]);

      // 读取相机1的内参
      std::vector<double> intrinsics1 = config["cameras"][1]["camera"]["intrinsics"]["data"].as<std::vector<double>>();
      k_seq_1 = (cv::Mat_<double>(1, 4) << intrinsics1[0], intrinsics1[1], intrinsics1[2], intrinsics1[3]);

      // 读取相机1的畸变参数
      std::vector<double> distortion1 = config["cameras"][1]["camera"]["distortion"]["parameters"]["data"].as<std::vector<double>>();
      D1 = (cv::Mat_<double>(1, 4) << distortion1[0], distortion1[1], distortion1[2], distortion1[3]);
      // 其他参数的读取...

      std::cout << "D1" << std::endl;
    }
    catch (YAML::Exception &e)
    {
      std::cerr << "YAML 异常: " << e.what() << std::endl;
    }
    catch (std::exception &e)
    {
      std::cerr << "标准异常: " << e.what() << std::endl;
    }
    catch (...)
    {
      std::cerr << "未知异常." << std::endl;
    }
    cv::Mat k_cam0 = (cv::Mat_<double>(3, 3) << k_seq_0.at<double>(0, 0), 0, k_seq_0.at<double>(0, 2), 0, k_seq_0.at<double>(0, 1), k_seq_0.at<double>(0, 3), 0, 0, 1);
    cv::Mat k_cam1 = (cv::Mat_<double>(3, 3) << k_seq_1.at<double>(0, 0), 0, k_seq_1.at<double>(0, 2), 0, k_seq_1.at<double>(0, 1), k_seq_1.at<double>(0, 3), 0, 0, 1); // 两个相机的内参
    cv::Mat r_w_cam0(3, 3, CV_64FC1);
    cv::Mat t_w_cam0(3, 1, CV_64FC1);
    cv::Mat r_w_cam1(3, 3, CV_64FC1);
    cv::Mat t_w_cam1(3, 1, CV_64FC1);
    cv::Mat z_cam0(3, 1, CV_64FC1);
    cv::Mat z_cam1(3, 1, CV_64FC1);
    region2Mat(r_w_cam0, pose_w_cam0, 3, 4, 0, 0, 3, 3);
    region2Mat(t_w_cam0, pose_w_cam0, 3, 4, 0, 3, 3, 1);
    region2Mat(z_cam0, pose_w_cam0, 3, 4, 0, 2, 3, 1);
    region2Mat(r_w_cam1, pose_w_cam1, 3, 4, 0, 0, 3, 3);
    region2Mat(t_w_cam1, pose_w_cam1, 3, 4, 0, 3, 3, 1);
    region2Mat(z_cam1, pose_w_cam1, 3, 4, 0, 2, 3, 1);

    cv::Mat baseline = t_w_cam1 - t_w_cam0; // 得到基线的向量，求取一个垂直于基线的平面

    cv::Mat new_x = baseline / norm(baseline); // 得到新的x轴，长度为1
    cv::Mat middle = (z_cam0 + z_cam1) / 2;    // 取两个坐标轴的中间值，将该点投影到平面
    middle = middle / norm(middle);
    cv::Mat middle_projection_2_plane = middle.t() * new_x; // 点乘，两个值需要长度都为1
    // ROS_INFO_STREAM("CAMERA"<<'\n'<<z_cam0 <<'\n'<<z_cam0<<"\n"<<middle_projection_2_plane);
    cv::Mat new_z = middle - middle_projection_2_plane.at<double>(0, 0) * new_x; // 获得为归一化的新z轴
    new_z = new_z / norm(new_z);
    // ROS_INFO_STREAM("middle_projection_2_plane"<<middle_projection_2_plane);
    // ROS_INFO_STREAM("middle"<<middle);
    // ROS_INFO_STREAM("new_x"<<new_x);
    // ROS_INFO_STREAM("new_z"<<new_z);

    cv::Mat new_y;
    cross(new_z, new_x, new_y);
    new_y = new_y / norm(new_y); // 获得y轴
    // ROS_INFO_STREAM("norm"<<new_x.mul(new_y)<<new_x.mul(new_z)<<new_z.mul(new_y));
    std::vector<cv::Mat> vImgs;
    cv::Mat result;
    vImgs.push_back(new_x);
    vImgs.push_back(new_y);
    vImgs.push_back(new_z);
    hconcat(vImgs, result); // 存储的是新平面的世界坐标系下的位姿
    // ROS_INFO_STREAM("new pose"<<result);//得到一个新的位姿矩阵
    //  ROS_INFO_STREAM("norm(new_x.mul(new_z))"<<norm(new_x.mul(new_z)));
    //  ROS_INFO_STREAM("norm(new_x.mul(new_y))"<<norm(new_x.mul(new_y)));
    //  ROS_INFO_STREAM("norm(new_y.mul(new_z))"<<norm(new_y.mul(new_z)));
    R0 = result.t() * r_w_cam0; // 相机1到新平面的旋转
    R1 = result.t() * r_w_cam1;
    // ROS_INFO_STREAM("result"<<result<<" "<<result.t());
    cv::Mat diagm = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    // R0=diagm;
    // R1=diagm;}
    vImgs.clear();
    vImgs.push_back(diagm); // 新平面下的世界坐标位姿
    vImgs.push_back(cv::Mat::zeros(3, 1, CV_64FC1));
    hconcat(vImgs, P0);
    P0 = k_cam0 * P0;
    P0.copyTo(P1);
    P1.at<double>(0, 3) = -k_cam0.at<double>(0, 0) * norm(t_w_cam1 - t_w_cam0);
    // ROS_INFO_STREAM("R0"<<R0<<"\nR1"<<R1);
    // ROS_INFO_STREAM("P0"<<P0<<"\nP1"<<P1);

    // Convert OpenCV image to ROS message
    pcam0 = getCameraInfo(k_cam0, D0, R0, P0, model_dist, 480, 640);
    pcam1 = getCameraInfo(k_cam1, D1, R1, P1, model_dist, 480, 640);
    camera.push_back(pcam0);
    camera.push_back(pcam1);
  }
  void dynamic_reconfigure_callback(stereo::dynamicConfig &config, uint32_t level)
  {

    // this->left_matcher=StereoSGBM::create(0, max_disp, wsize);
    BlockSize = (config.BlockSize);
    p1_num = (config.p1_num);
    p2_num = (config.p2_num);
    Disp12MaxDiff = (config.Disp12MaxDiff);
    MinDisparity = (config.MinDisparity);
    SpeckleWindowSize = (config.SpeckleWindowSize);
    SpeckleRange = (config.SpeckleRange);
    PreFilterCap = (config.PreFilterCap);
    UniquenessRatio = (config.UniquenessRatio);
    sgbm_type = (static_cast<decltype(StereoSGBM::MODE_SGBM_3WAY)>(config.sgbm_type));
  };
  void region2Mat(cv::Mat &P, const cv::Mat seq, int seq_rows, int seq_cols, int t, int l, int mat_rows, int mat_cols)
  {
    if (l != -1)
    {
      int idx = 0;
      int re = l + mat_cols - 1;
      for (int i = t * seq_cols + l; i < seq_rows * seq_cols; i++)
      {
        if (i % seq_cols < l)
        {
          i = (i - i % seq_cols) + l;
        }
        if (i % seq_cols > re)
        {
          i = (i / seq_cols + 1) * seq_cols + l;
        }
        if (idx >= mat_cols * mat_rows)
        {
          return;
        }
        P.at<double>(idx / mat_cols, idx % mat_cols) = seq.at<double>(i / seq_cols, i % seq_cols);
        idx++;
      }
    }
    else
    {
      for (int i = 0; i < seq_rows * seq_cols; i++)
      {
        P.at<double>(i / seq_cols, i % seq_cols) = seq.at<double>(i / seq_cols, i % seq_cols);
      }
    }
  }

  void invert_pose(cv::Mat &out_r, cv::Mat &out_t, const cv::Mat &r, const cv::Mat &t)
  {
    out_r = r.t();
    out_t = -t;
  }

  void cross(cv::Mat &out_r, cv::Mat &out_t, cv::Mat &dest)
  {
    cv::Mat hat_mat = (cv::Mat_<double>(3, 3) << 0, -1 * out_r.at<double>(2, 0), out_r.at<double>(1, 0), out_r.at<double>(2, 0), 0, -1 * out_r.at<double>(0, 0), -1 * out_r.at<double>(1, 0), out_r.at<double>(0, 0), 0);
    dest = hat_mat * out_t;
  }

  sensor_msgs::CameraInfo getCameraInfo(cv::Mat k, cv::Mat d, cv::Mat r, cv::Mat p, std::string model, int h, int w)
  {
    sensor_msgs::CameraInfo cam;
    std::vector<double> D{d.at<double>(0), d.at<double>(1), d.at<double>(2), d.at<double>(3), 0};
    boost::array<double, 9> K = {
        k.at<double>(0, 0),
        k.at<double>(0, 1),
        k.at<double>(0, 2),
        k.at<double>(1, 0),
        k.at<double>(1, 1),
        k.at<double>(1, 2),
        k.at<double>(2, 0),
        k.at<double>(2, 1),
        k.at<double>(2, 2),
    };

    // get rectified projection.
    boost::array<double, 12> P = {
        p.at<double>(0, 0), p.at<double>(0, 1), p.at<double>(0, 2), p.at<double>(0, 3),
        p.at<double>(1, 0), p.at<double>(1, 1), p.at<double>(1, 2), p.at<double>(1, 3),
        p.at<double>(2, 0), p.at<double>(2, 1), p.at<double>(2, 2), p.at<double>(2, 3)};
    boost::array<double, 9> R = {
        r.at<double>(0, 0),
        r.at<double>(0, 1),
        r.at<double>(0, 2),
        r.at<double>(1, 0),
        r.at<double>(1, 1),
        r.at<double>(1, 2),
        r.at<double>(2, 0),
        r.at<double>(2, 1),
        r.at<double>(2, 2),
    };

    cam.height = h;
    cam.width = w;
    cam.distortion_model = model;
    cam.D = D;
    cam.K = K;
    cam.P = P;
    cam.R = R;
    cam.binning_x = 1;
    cam.binning_y = 1;
    return cam;
  }
  void
  compute_stereo(Mat &imL, Mat &imR, image_geometry::StereoCameraModel model_)
  {
    Mat Limage(imL);
    // confidence map
    Mat conf_map = Mat(imL.rows, imL.cols, CV_8U);
    conf_map = Scalar(255);

    // downsample images to speed up results

    resize(imL, left_for_matcher, Size(), 1, 1);
    resize(imR, right_for_matcher, Size(), 1, 1);

    max_disp = 16 * 10;
    // max_disp /= 2;
    if (max_disp % 16 != 0)
      max_disp += 16 - (max_disp % 16);
    Ptr<StereoSGBM> left_matcher = StereoSGBM::create(0, max_disp, BlockSize);
    left_matcher->setBlockSize(BlockSize);
    left_matcher->setP1(p1_num * BlockSize * BlockSize);
    left_matcher->setP2(p2_num * BlockSize * BlockSize);

    left_matcher->setDisp12MaxDiff(Disp12MaxDiff);
    left_matcher->setMinDisparity(MinDisparity);
    left_matcher->setSpeckleWindowSize(SpeckleWindowSize);
    left_matcher->setSpeckleRange(SpeckleRange);
    left_matcher->setPreFilterCap(PreFilterCap);
    left_matcher->setUniquenessRatio(UniquenessRatio);
    left_matcher->setMode(static_cast<decltype(StereoSGBM::MODE_SGBM_3WAY)>(sgbm_type));
    Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);
    wls_filter = createDisparityWLSFilter(left_matcher);
    left_matcher->compute(left_for_matcher, right_for_matcher, left_disp);
    right_matcher->compute(right_for_matcher, left_for_matcher, right_disp);

    // filter
    wls_filter->setLambda(lambda);
    wls_filter->setSigmaColor(sigma);
    wls_filter->filter(left_disp, imL, filtered_disp, right_disp);
    conf_map = wls_filter->getConfidenceMap();
    ROI = wls_filter->getROI();

    // visualization
    getDisparityVis(filtered_disp, filtered_disp_vis, vis_mult);
    // filtered_disp=filtered_disp/2;
    // PointCloud Generation======================================================

    // Q matrix (guess until we can do the correct calib process)
    // double w = imR.cols;
    // double  h = imR.rows;
    // double f = 843.947693;
    // double cx = 508.062911;
    // double cx1 = 526.242457;
    // double cy = 385.070250;
    // double Tx = -120.00;
    double w = imR.cols;
    double  h = imR.rows;
    double f = 382.34175236928627;
    double cx = 321.7242056112083;
    double cx1 = 322.3064389150983;
    double cy = 236.24691075834372;
    double Tx = -19.29692482;
    Mat Q = Mat(4,4, CV_64F, double(0));
    Q.at<double>(0,0) = 1.0;
    Q.at<double>(0,3) = -cx;
    Q.at<double>(1,1) = 1.0;
    Q.at<double>(1,3) = -cy;
    Q.at<double>(2,3) = f;
    Q.at<double>(3,2) = -1.0/ Tx;
    Q.at<double>(3,3) = ( cx - cx1)/ Tx;
    //         [ 1 0   0      -Cx      ]
    // Q = [ 0 1   0      -Cy      ]
    //     [ 0 0   0       Fx      ]
    //     [ 0 0 -1/Tx (Cx-Cx')/Tx ]
    // double Tx = -19.42830883756359;
    // cv::Mat Q = (cv::Mat_<double>(4, 4) << -19.42830883756359, 0, 0, 6251.133074507617,
    //              0, -19.42830883756359, 0, 4592.437057166097,
    //              0, 0, 0, -7472.829104043432,
    //              0, 0, -1, 11.40046179682074 / 384.6361083984375);
    // Q = Q * Tx;
    std::cout << Q << std::endl;
  
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    Mat xyz;
    this->model_.projectDisparityImageTo3d(filtered_disp, xyz, true);
    // reprojectImageTo3D(filtered_disp, xyz, Q, true);
    pointcloud->width = static_cast<uint32_t>(filtered_disp.cols);
    pointcloud->height = static_cast<uint32_t>(filtered_disp.rows);
    pointcloud->is_dense = true;
    pcl::PointXYZRGB point;
    for (int i = 0; i < filtered_disp.rows; ++i)
    {
      uchar *rgb_ptr = Limage.ptr<uchar>(i);
      uchar *filtered_disp_ptr = filtered_disp.ptr<uchar>(i);
      double *xyz_ptr = xyz.ptr<double>(i);

      for (int j = 0; j < filtered_disp.cols; ++j)
      {

        uchar d = filtered_disp_ptr[j];
        // if (d == 0) continue;
        Point3f p = xyz.at<Point3f>(i, j);

        double radius = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        if (radius < 20 * 100)
        {
          point.z = p.z *19.42830883756359;
          point.x = p.x *19.42830883756359;
          point.y = p.y *19.42830883756359;
          point.b = rgb_ptr[j];
          point.g = rgb_ptr[j];
          point.r = rgb_ptr[j];
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
    pcl::toROSMsg(*pointcloud, pointcloud_msg);
    pointcloud_msg.header.frame_id = "cam_pos";

    // Publishes pointcloud message
    pcpub.publish(pointcloud_msg);
  }

  void callback(const ImageConstPtr &left, const ImageConstPtr &right)
  {
    // conversion to rosmsgs::Image to cv::Mat using cv_bridge

    cv_bridge::CvImagePtr cv_left;
    try
    {
      cv_left = cv_bridge::toCvCopy(left);
    }
    catch (cv_bridge::Exception &e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    cv_bridge::CvImagePtr cv_right;
    try
    {
      cv_right = cv_bridge::toCvCopy(right);
    }
    catch (cv_bridge::Exception &e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    compute_stereo(cv_left->image, cv_right->image, this->model_);
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", filtered_disp_vis).toImageMsg();
    pub.publish(msg);
  }

private:
  ros::NodeHandle pnh;
  bool enableVisualization = 0;
  image_transport::Publisher pub;
  ros::Publisher pcpub;
  // parameters for stereo matching and filtering
  double vis_mult = 5.0;
  int BlockSize = 13;
  int max_disp = 16 * 10;
  double lambda = 10000.0;
  double sigma = 1.0;
  // Some object instatiation that can be done only once
  Mat left_for_matcher, right_for_matcher;
  Mat left_disp, right_disp;
  Mat filtered_disp;
  Rect ROI;
  Ptr<DisparityWLSFilter> wls_filter;
  Ptr<StereoSGBM> left_matcher;
  Mat filtered_disp_vis;
  int sgbm_type;
  int p1_num, p2_num, prefilter_num;
  int Disp12MaxDiff, MinDisparity, SpeckleWindowSize, SpeckleRange, PreFilterCap, UniquenessRatio;
  dynamic_reconfigure::Server<stereo::dynamicConfig> server;
  image_transport::ImageTransport it;
  boost::shared_ptr<message_filters::Subscriber<Image>> left_sub;
  boost::shared_ptr<message_filters::Subscriber<Image>> right_sub;

  // time syncronizer to publish 2 images in the same callback function
  image_geometry::StereoCameraModel model_;
  boost::shared_ptr<TimeSynchronizer<Image, Image>> sync;
};

int main(int argc, char **argv)
{

  ros::init(argc, argv, "stereo_node_");
  ros::NodeHandle nh;
  stereo_node_ mynode(nh);
  ros::spin();
  return 0;
}
