#include "realsense.h"

Realsense::Realsense()
{
    img_arrived_.store(false,std::memory_order_acquire);
}

Realsense::~Realsense()
{

}

bool Realsense::setupInputDevice(const InputDevSetup& setup)
{
    dev_setup_ = setup;
    config_.enable_stream(RS2_STREAM_COLOR,dev_setup_.width,dev_setup_.height,RS2_FORMAT_BGR8,dev_setup_.fps);
    config_.enable_stream(RS2_STREAM_DEPTH,dev_setup_.width,dev_setup_.height,RS2_FORMAT_Z16,dev_setup_.fps);

    #ifdef D435i
    config_.enable_stream(RS2_STREAM_ACCEL,RS2_FORMAT_MOTION_XYZ32F);
    config_.enable_stream(RS2_STREAM_GYRO,RS2_FORMAT_MOTION_XYZ32F);
    #endif
    
    auto imuCallback_ = [&](rs2::frame frame)
    {
        if(rs2::frameset fs = frame.as<rs2::frameset>())
        {
            img_mutex_.lock();
            frameset = fs;
            img_mutex_.unlock();

            img_arrived_.store(true,std::memory_order_acquire);
        }

        else if(rs2::motion_frame m_frame = frame.as<rs2::motion_frame>())
        {
            rs2::motion_frame motion = frame.as<rs2::motion_frame>();

            if(motion && motion.get_profile().stream_type() == RS2_STREAM_ACCEL && motion.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F)
            {
                imu_mutex_.lock();
                accel_data_ = motion.get_motion_data();
                imu_mutex_.unlock();
            }
            
            else if(motion && motion.get_profile().stream_type() == RS2_STREAM_GYRO && motion.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F)
            {
                imu_mutex_.lock();
                gyro_data_ = motion.get_motion_data();
                imu_mutex_.unlock();
            }
        }
    };

    selection_ = pipe_.start(config_,imuCallback_);
    dev_params_.setDefault();

    return true;
    
}

bool Realsense::startStreaming()
{
    while(!img_arrived_.load(std::memory_order_acquire))
    {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    img_mutex_.lock();
    rs2::video_frame colour_frame = frameset.get_color_frame();
    rs2::depth_frame depth = frameset.get_depth_frame();
    img_mutex_.unlock();

    depth_frame_.data = cv::Mat(cv::Size(dev_setup_.width,dev_setup_.height),CV_16U,(void*)depth.get_data(),cv::Mat::AUTO_STEP);
    colour_frame_.bgr_frame = cv::Mat(cv::Size(dev_setup_.width,dev_setup_.height),CV_8UC3,(void*)colour_frame.get_data());

    img_arrived_.store(false,std::memory_order_acquire);

    return true;
}

bool Realsense::getIMUData(IMUFrame& data)
{
    imu_mutex_.lock();
    data.accel.x = accel_data_.z;
    data.accel.y = -accel_data_.x;
    data.accel.z = -accel_data_.y;

    data.gyro.x = gyro_data_.z;
    data.gyro.y = -gyro_data_.x;
    data.gyro.z = -gyro_data_.y;

    data.gyro_time = gyro_time_;
    data.accel_time = accel_time_;
    imu_mutex_.unlock();
    return true;
}

bool Realsense::getColourFrame(ColourFrame& frame)
{
    colour_frame_.BGR2Gray();
    frame = colour_frame_;
    return true;
}

bool Realsense::getDepthFrame(DepthFrame& frame)
{
    depth_frame_.computeDepth(dev_setup_.depth_map_factor);
    frame = depth_frame_;
    return true;
}

CameraParams& Realsense::getDevParams()
{
    return dev_params_;
}
