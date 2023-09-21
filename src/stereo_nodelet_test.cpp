#include <pluginlib/class_list_macros.h>
#include <nodelet/nodelet.h>
#include "../include/stereoMagic/stereo_nodelet_test.h"

// PLUGINLIB_EXPORT_CLASS()将C++类制作为为ros plugin。
PLUGINLIB_EXPORT_CLASS(magicStereo::magic_stereo_nodelet_test, nodelet::Nodelet)

namespace magicStereo
{

    void magic_stereo_nodelet_test::onInit()
    {
        NODELET_DEBUG("Init nodelet...");
        ros::NodeHandle nh(getNodeHandle());
        ros::NodeHandle pnh(getPrivateNodeHandle());
        ros::NodeHandle nnh("");
        image_transport::ImageTransport it(nh);

        // pointcloud publisher
        // pcpub = pnh.advertise<sensor_msgs::PointCloud2>("/camera/depth/pointcloud", 1);
        // // depth image publisher
        // pub = it.advertise("/camera/depth/image", 1);

        // pnh.param<std::string>("left_image_topic",left_image_topic,"/camera/infra1/image_rect_raw");
        // pnh.param<std::string>("right_image_topic",right_image_topic,"/camera/infra1/image_rect_raw");
        // typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image> ExactPolicy;
        // typedef message_filters::Synchronizer<ExactPolicy> ExactSync;
        // image_transport::SubscriberFilter sub0(it, left_image_topic, 1, std::string("raw"));
        // image_transport::SubscriberFilter sub1(it, right_image_topic, 1, std::string("raw"));
        // ExactSync sync_sub(ExactPolicy(5), sub0, sub1);
        // sync_sub.registerCallback(boost::bind(&magicStereo::magic_stereo_nodelet_test::callback, this, _1, _2));

//   image_transport::Subscriber left_sub_=it.subscribe( "/camera/infra1/image_rect_raw", 1,boost::bind<void ,const sensor_msgs::ImageConstPtr&>(imageCallback,_1));
        depth_thread_.reset(new boost::thread(&magicStereo::magic_stereo_nodelet_test::depthLoop, this,nh));
        
        //ros::spin();
    }
    void magic_stereo_nodelet_test::depthLoop(ros::NodeHandle &nh_)
    {
        ros::NodeHandle nh(nh_, "depthLoop");

        ros::CallbackQueue queue;
        nh.setCallbackQueue(&queue);//设置回调函数队列，并且spin或spinonce不会调用这个队列里的函数，需要自己处理调用队列内函数的操作

        // subscribe to cam msgs
        image_transport::ImageTransport it(nh);
        image_transport::Subscriber depth_sub=it.subscribe("/camera/depth/image",1,boost::bind(&magicStereo::magic_stereo_nodelet_test::depthCallback,this,_1));

        while (ros::ok())
        {
            queue.callAvailable(ros::WallDuration(0.1));
        }
    }
    void magic_stereo_nodelet_test::depthCallback(const sensor_msgs::ImageConstPtr &depth)
    {
       ROS_INFO_STREAM("subscribe address is"<<depth<<"  data address is "<<&(depth->data));
    }
    void magic_stereo_nodelet_test::compute_stereo(cv::Mat &imL, cv::Mat &imR)
    {
        cv::Mat Limage(imL);
        // confidence map74546
        cv::Mat conf_map = cv::Mat(imL.rows, imL.cols, CV_8U);
        conf_map = cv::Scalar(255);

        // downsample images to speed up results
        max_disp /= 2;
        if (max_disp % 16 != 0)
            max_disp += 16 - (max_disp % 16);
        resize(imL, left_for_matcher, cv::Size(), 0.5, 0.5);
        resize(imR, right_for_matcher, cv::Size(), 0.5, 0.5);

        // compute disparity
        cv::Ptr<cv::StereoSGBM> left_matcher = cv::StereoSGBM::create(0, max_disp, wsize);
        left_matcher->setP1(24 * wsize * wsize);
        left_matcher->setP2(96 * wsize * wsize);
        left_matcher->setPreFilterCap(63);
        left_matcher->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
        wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);
        cv::Ptr<cv::StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(left_matcher);

        left_matcher->compute(left_for_matcher, right_for_matcher, left_disp);
        right_matcher->compute(right_for_matcher, left_for_matcher, right_disp);

        // filter
        wls_filter->setLambda(lambda);
        wls_filter->setSigmaColor(sigma);
        wls_filter->filter(left_disp, imL, filtered_disp, right_disp);
        conf_map = wls_filter->getConfidenceMap();
        ROI = wls_filter->getROI();

        // visualization
        cv::ximgproc::getDisparityVis(filtered_disp, filtered_disp_vis, vis_mult);

        // PointCloud Generation======================================================
        double w = imR.cols;
        double h = imR.rows;
        double f = 382.34175236928627;
        double cx = 321.7242056112083;
        double cx1 = 322.3064389150983;
        double cy = 236.24691075834372;
        double Tx = -19.29692482;
        cv::Mat Q = cv::Mat(4, 4, CV_64F, double(0));
        Q.at<double>(0, 0) = 1.0;
        Q.at<double>(0, 3) = -cx;
        Q.at<double>(1, 1) = 1.0;
        Q.at<double>(1, 3) = -cy;
        Q.at<double>(2, 3) = f;
        Q.at<double>(3, 2) = -1.0 / Tx;
        Q.at<double>(3, 3) = (cx - cx1) / Tx;

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        cv::Mat xyz;
        reprojectImageTo3D(filtered_disp, xyz, Q, true);
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
                cv::Point3f p = xyz.at<cv::Point3f>(i, j);

                double radius = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
                if (radius < 20 * 100)
                {
                    point.z = p.z / 100.0;
                    point.x = p.x / 100.0;
                    point.y = p.y / 100.0;
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
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::VoxelGrid<pcl::PointXYZRGB> sor;
        sor.setInputCloud(pointcloud);
        sor.setLeafSize(0.01, 0.01, 0.01);
        sor.filter(*cloud_filtered);

        // outliner removal filter
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered2(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor1;
        sor1.setInputCloud(cloud_filtered);
        sor1.setMeanK(100);
        sor1.setStddevMulThresh(0.001);
        sor1.filter(*cloud_filtered2);
        if (enableVisualization)
        {
            pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
            cv::imshow(OPENCV_WINDOW, filtered_disp_vis);
            viewer.showCloud(cloud_filtered2);
            cv::waitKey(3);
        }

        // Convert to ROS data type
        sensor_msgs::PointCloud2 pointcloud_msg;
        pcl::toROSMsg(*cloud_filtered2, pointcloud_msg);
        pointcloud_msg.header.frame_id = "cam_pos";

        // Publishes pointcloud message
        pcpub.publish(pointcloud_msg);
    };

    void magic_stereo_nodelet_test::callback(const sensor_msgs::ImageConstPtr &left, const sensor_msgs::ImageConstPtr &right)
    {
        // conversion to rosmsgs::Image to cv::Mat using cv_bridge
        if (left == nullptr)
            ROS_INFO_STREAM("empty image ptr");
        else
            ROS_INFO_STREAM("get image and callback");
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

        compute_stereo(cv_left->image, cv_right->image);
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", filtered_disp_vis).toImageMsg();
        pub.publish(msg);
    };
}
