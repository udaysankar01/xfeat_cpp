#ifndef FRAMES_H
#define FRAMES_H

#include "opencv2/opencv.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include <opencv2/core/types_c.h>
#include <opencv2/core/core_c.h>
#include "opencv2/ximgproc.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <eigen3/Eigen/Dense>
#include <librealsense2/rs.hpp>

#define DEFAULT_CAM_HEIGHT 480
#define DEFAULT_CAM_WIDTH 640
#define FPS 30

#define CAM_Fx 617.201
#define CAM_Fy 617.362
#define CAM_Cx 324.637
#define CAM_Cy 242.462

#define CAM_K1 0
#define CAM_K2 0
#define CAM_P1 0
#define CAM_P2 0
#define CAM_K3 0
#define DEPTH_MAP_FACTOR 1000
#define CAM_BASE_LENGTH 0.07732

/**
 * @brief Camera Frame
 * 
 */
typedef struct 
{
    cv::Mat bgr_frame;
    cv::Mat gray_frame;
    double time = 0;

    void BGR2Gray()
    {
        cv::cvtColor(bgr_frame,gray_frame,cv::COLOR_BGR2GRAY);
    }
    
    void vis(std::string name)
    {
        cv::imshow(name,gray_frame);
        cv::waitKey(10);
    }

    void destroy(std::string name)
    {
        cv::destroyWindow(name);
    }

}ColourFrame;

/**
 * @brief Depth Frame
 * 
 */
typedef struct
{
    cv::Mat data;

    void computeDepth(double scale)
    {
        cv::Mat tmp_depth;
        data.convertTo(tmp_depth,CV_64F,scale);
        data = tmp_depth.clone();
    }

    double getDepth(int u,int v)
    {
        return data.at<double>(u,v);
    }

}DepthFrame;

/**
 * @brief Camera parameters 
 * 
 */
typedef struct 
{
    cv::Mat intrinsics = cv::Mat::zeros(cv::Size(3,3),CV_64FC1);
    cv::Mat dist_coeff = cv::Mat::zeros(cv::Size(5,1),CV_64FC1);
    double base_length = 0;
    
    
    void setDefault()
    {
        intrinsics.at<double>(0,0) = CAM_Fx;
        intrinsics.at<double>(0,1) = 0;
        intrinsics.at<double>(0,2) = CAM_Cx;
        intrinsics.at<double>(1,0) = 0;
        intrinsics.at<double>(1,1) = CAM_Fy;
        intrinsics.at<double>(1,2) = CAM_Cy;
        intrinsics.at<double>(2,0) = 0;
        intrinsics.at<double>(2,1) = 0;
        intrinsics.at<double>(2,2) = 1;

        dist_coeff.at<double>(0) = CAM_K1;
        dist_coeff.at<double>(1) = CAM_K2;
        dist_coeff.at<double>(2) = CAM_P1;
        dist_coeff.at<double>(3) = CAM_P2;
        dist_coeff.at<double>(4) = CAM_K3;

        base_length = CAM_BASE_LENGTH;
    }

}CameraParams;

/**
 * @brief Input device setup parameters
 * 
 */
typedef struct 
{
    int height;
    int width;
    int fps;
    double depth_map_factor;
    std::string serial_num_;
    
    void setDefault()
    {
        height = DEFAULT_CAM_HEIGHT;
        width = DEFAULT_CAM_WIDTH;
        fps = FPS;
        depth_map_factor = 1.0 / DEPTH_MAP_FACTOR;
    }

}InputDevSetup;

typedef struct 
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
}SensorData;

typedef struct
{
    SensorData gyro;
    SensorData accel;

    double gyro_time;
    double accel_time;
}IMUFrame;



#endif