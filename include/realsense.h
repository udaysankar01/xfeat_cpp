#ifndef REALSENSE_H
#define REALSENSE_H

#include "frames.h"
#include "thread"
#include "mutex"
#include "atomic"

#define D435i //comment this if your camera does not have IMU

class Realsense 
{
    public:
        /**
         * @brief Construct a new Realsense object
         * 
         */
        Realsense();
        /**
         * @brief Destroy the Realsense object
         * 
         */
        ~Realsense();

        /**
         * @brief 
         * 
         * @param setup 
         * @return true 
         * @return false 
         */
        bool setupInputDevice(const InputDevSetup& setup);

        /**
         * @brief 
         * 
         * @return true 
         * @return false 
         */
        bool startStreaming();

        /**
         * @brief Get the Colour Frame object
         * 
         * @param frame 
         * @return true 
         * @return false 
         */
        bool getColourFrame(ColourFrame& frame);
        
        /**
         * @brief Get the Depth Frame object
         * 
         * @param depth 
         * @return true 
         * @return false 
         */
        bool getDepthFrame(DepthFrame& depth);

        /**
         * @brief 
         * 
         * @param imu 
         * @return true 
         * @return false 
         */
        bool getIMUData(IMUFrame& imu);

        /**
         * @brief Get the Dev Params object
         * 
         * @return CameraParams& 
         */
        CameraParams& getDevParams();

    private:
        /**
         * @brief 
         * 
         */
        InputDevSetup dev_setup_;
        /**
         * @brief 
         * 
         */
        ColourFrame colour_frame_;
        /**
         * @brief 
         * 
         */
        DepthFrame depth_frame_;
        /**
         * @brief 
         * 
         */
        CameraParams dev_params_;
        /**
         * @brief 
         * 
         */
        rs2::pipeline pipe_;
        /**
         * @brief 
         * 
         */
        rs2::config config_;
        /**
         * @brief 
         * 
         */
        rs2::pipeline_profile selection_;
        /**
         * @brief 
         * 
         */
        std::mutex img_mutex_;
        /**
         * @brief 
         * 
         */
        std::mutex imu_mutex_;
        /**
         * @brief 
         * 
         */
        rs2::frameset frameset;
        /**
         * @brief 
         * 
         */
        std::atomic<bool> img_arrived_;
        /**
         * @brief 
         * 
         */
        rs2_vector accel_data_;
        /**
         * @brief 
         * 
         */
        rs2_vector gyro_data_;
        /**
         * @brief 
         * 
         */
        double accel_time_;
        /**
         * @brief 
         * 
         */
        double gyro_time_;

};

#endif
