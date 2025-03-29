#include "realsense.h"
#include "XFeat.h"
#include "mutex"
#include "thread"
#include "unistd.h"


char val = 'q';
std::mutex mut;

void keyboardInputFunction()
{
    while(1)
    {
        std::cout<<"Press s to set the reference frame: "<<std::endl;
        
        char tmp;
        std::cin>>tmp;

        mut.lock();
        val = tmp;
        mut.unlock();

        std::cout<<"reference image set"<<std::endl;
        usleep(100);
    }
}

int main()
{
    std::thread input_thread(keyboardInputFunction);
    int top_k = 1000;
    float detection_threshold = 0.5;
    bool use_cuda = false; 
    XFeat::XFDetector detector(top_k, detection_threshold, use_cuda);

    Realsense realsense_camera;
    InputDevSetup camera_setup;
    camera_setup.setDefault();
    realsense_camera.setupInputDevice(camera_setup);
    bool first_img_set = false;

    cv::Mat ref_img;
    cv::Mat ref_desc;
    std::vector<cv::KeyPoint> ref_kps;

    while(1)
    {
        ColourFrame frame;

        realsense_camera.startStreaming();
        realsense_camera.getColourFrame(frame);
        frame.vis("camera feed");
        char tmp;

        mut.lock();
        tmp = val;
        mut.unlock();

        if(tmp == 's')
        {
            mut.lock();
            val = 'q';
            mut.unlock();

            ref_img = frame.bgr_frame.clone();
            first_img_set = true;
            detector.extractFeatures(ref_img,ref_kps,ref_desc);
            continue;
        }

        if(!first_img_set)
        {
            continue;
        }

        cv::imshow("reference image",ref_img);

        cv::Mat desc;
        std::vector<cv::KeyPoint> kps;
        std::vector<cv::DMatch> matches;

        detector.extractFeatures(frame.bgr_frame,kps,desc);
        detector.matchFeatures(ref_desc,desc,matches);

        cv::Mat output_img;
        cv::drawMatches(ref_img,ref_kps,frame.bgr_frame,kps,matches,output_img);

        cv::imshow("matching frame",output_img);
    }   

    input_thread.join();
    return 1;
}