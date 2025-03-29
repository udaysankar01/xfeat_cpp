#include "XFeat.h"
#include <thread>
#include <atomic>
#include <chrono>
#include <vector>
#include <mutex>

using namespace XFeat;

struct Args 
{
    int width = 640;
    int height = 480;
    int max_kpts = 1000;
    int cam = 0;
};

class FrameGrabber 
{
public:
    FrameGrabber(cv::VideoCapture& cap) 
        : cap(cap), running(false) 
    {
        cap.read(frame);
    }

    void start() {
        running = true;
        grabberThread = std::thread(&FrameGrabber::run, this);
        if (grabberThread.joinable()) {
            std::cout << "Frame grabber thread started successfully." << std::endl;
        } else {
            std::cerr << "Failed to start frame grabber thread." << std::endl;
        }
    }

    void stop() {
        running = false;
        if (grabberThread.joinable()) {
            grabberThread.join();
            std::cout << "Frame grabber thread stopped successfully." << std::endl;
        }
        cap.release();
    }

    cv::Mat get_last_frame() {
        std::lock_guard<std::mutex> lock(frameMutex);
        return frame.clone();
    }

private:
    void run()
    {
        running = true;
        while (running)
        {
            bool ret = cap.read(frame);
            if (!ret)
            {
                std::cout << "Can't receive frame (stream ended?).\n";
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    cv::VideoCapture& cap;
    cv::Mat frame;
    std::atomic<bool> running;
    std::thread grabberThread;
    std::mutex frameMutex;
};

class MatchingDemo 
{
public:
    MatchingDemo(const Args& args)
        : args(args),
          cap(args.cam, cv::CAP_V4L2),
          frameGrabber(cap)
    {
        setupCamera();
        frameGrabber.start();
        cv::namedWindow(windowName, cv::WINDOW_GUI_NORMAL);
        cv::resizeWindow(windowName, args.width * 2, args.height * 2);
        cv::setMouseCallback(windowName, mouseCallback, this);

        xfeatDetector = std::make_shared<XFDetector>(args.max_kpts, 0.1, true);
    }

    ~MatchingDemo() 
    {
        cleanup();
    }

    void main_loop() 
    {
        currentFrame = frameGrabber.get_last_frame();
        refFrame = currentFrame.clone();
        auto refFrameTensor =  xfeatDetector->parseInput(refFrame);
        xfeatDetector->detectAndCompute(refFrameTensor, refKeypointsDescriptors);

        while (true) 
        {

            currentFrame = frameGrabber.get_last_frame();

            if (currentFrame.empty()) 
            {
                continue;
            }

            double t0 = cv::getTickCount();
            process();

            int key = cv::waitKey(1);
            if (key == 'q') 
                break;
            if (key == 's') 
            {
                refFrame = currentFrame.clone();
                auto inputTensor = xfeatDetector->parseInput(refFrame);
                xfeatDetector->detectAndCompute(inputTensor, refKeypointsDescriptors);
            }
            
            double t1 = cv::getTickCount();
            timeList.push_back((t1 - t0) / cv::getTickFrequency());
            if (timeList.size() > maxCnt) timeList.erase(timeList.begin());
            FPS = 1.0 / (std::accumulate(timeList.begin(), timeList.end(), 0.0) / timeList.size());
        }
    }

private:
    void setupCamera() 
    {
        cap.set(cv::CAP_PROP_FRAME_WIDTH, args.width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, args.height);
        cap.set(cv::CAP_PROP_FPS,  59);
        if (!cap.isOpened()) 
        {
            std::cerr << "Cannot open camera\n";
            exit(EXIT_FAILURE);
        }
    }

    void process() 
    {
        if (refFrame.empty() || currentFrame.empty()) 
        {
            return;
        }

        cv::Mat topFrameCanvas = createTopFrame();
        cv::Mat bottomFrame = matchAndDraw(refFrame, currentFrame);
        cv::Mat canvas;
        cv::vconcat(topFrameCanvas, bottomFrame, canvas);

        std::string fpsText = "FPS: " + std::to_string(FPS);
        cv::putText(canvas, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

        cv::imshow(windowName, canvas);
    }

    cv::Mat createTopFrame() 
    {
        if (refFrame.empty() || currentFrame.empty()) 
        {
            return cv::Mat(480, 1280, CV_8UC3, cv::Scalar(0, 0, 0));
        }

        cv::Mat topFrameCanvas(480, 1280, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::Mat topFrame;
        
        if (refFrame.rows == currentFrame.rows && refFrame.type() == currentFrame.type()) 
        {
            cv::hconcat(refFrame, currentFrame, topFrame);
        } 
        else 
        {
            return topFrameCanvas;
        }
        
        topFrame.copyTo(topFrameCanvas(cv::Rect(0, 0, topFrame.cols, topFrame.rows)));
        drawQuad(topFrameCanvas, corners);
        return topFrameCanvas;
    }

    cv::Mat matchAndDraw(cv::Mat& refFrame, cv::Mat& currentFrame) 
    {
        torch::Tensor current = xfeatDetector->parseInput(currentFrame);
        std::unordered_map<std::string, at::Tensor> currentKeypointsDescriptors;
        xfeatDetector->detectAndCompute(current, currentKeypointsDescriptors);

        auto out1 = refKeypointsDescriptors;
        auto out2 = currentKeypointsDescriptors;

        torch::Tensor idxs0, idxs1;
        double min_cossim = 0.82;
        xfeatDetector->match(out1["descriptors"], out2["descriptors"], idxs0, idxs1, min_cossim);

        torch::Tensor mkpts_0 = out1["keypoints"].index({idxs0});
        torch::Tensor mkpts_1 = out2["keypoints"].index({idxs1});
        cv::Mat mkpts_0_cv = xfeatDetector->tensorToMat(mkpts_0);
        cv::Mat mkpts_1_cv = xfeatDetector->tensorToMat(mkpts_1);

        if (mkpts_0_cv.rows < 4 && mkpts_1_cv.rows < 4)
        {
            std::cerr << "Not enough points to compute homography" << std::endl;
            exit(EXIT_FAILURE);
        }

        cv::Mat mask;
        cv::Mat H = cv::findHomography(mkpts_0_cv, mkpts_1_cv, cv::RANSAC, 10.0, mask, 1000, 0.994);
        if (H.empty()) 
        {
            std::cerr << "Homography matrix is empty" << std::endl;
            exit(EXIT_FAILURE);
        }
        else 
        {
            std::vector<cv::Point2f> projectedPoints;
            cv::perspectiveTransform(corners, projectedPoints, H);
            drawQuad(currentFrame, projectedPoints);
        }

        std::vector<cv::Point2f> points1, points2;
        std::vector<cv::KeyPoint> keypoints1, keypoints2;
        std::vector<cv::DMatch> matches;

        for (int i = 0; i < mkpts_0_cv.rows; ++i) 
        {   
            points1.emplace_back(mkpts_0_cv.at<float>(i, 0), mkpts_0_cv.at<float>(i, 1));
            points2.emplace_back(mkpts_1_cv.at<float>(i, 0), mkpts_1_cv.at<float>(i, 1));
            keypoints1.emplace_back(points1.back(), 1.0f);
            keypoints2.emplace_back(points2.back(), 1.0f);
            if (mask.at<uchar>(i, 0))
                matches.emplace_back(i, i, 0);
        }

        cv::Mat matchedFrame;
        cv::drawMatches(refFrame, keypoints1, currentFrame, keypoints2, matches, matchedFrame, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        
        // number of matches
        int numMatches = matches.size();
        std::string matchesText = "Matches: " + std::to_string(numMatches);
        cv::putText(matchedFrame, matchesText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        
        return matchedFrame;
    }

    static void drawQuad(cv::Mat& frame, const std::vector<cv::Point2f>& points) 
    {
        if (points.size() == 4) 
        {
            for (size_t i = 0; i < 4; ++i) 
            {
                cv::line(frame, points[i], points[(i + 1) % 4], cv::Scalar(0, 255, 0), 3);
            }
        }
    }

    static void mouseCallback(int event, int x, int y, int flags, void* userdata) 
    {
        if (event == cv::EVENT_LBUTTONDOWN) 
        {
            auto* demo = static_cast<MatchingDemo*>(userdata);
            if (demo->corners.size() >= 4) demo->corners.clear();
            demo->corners.emplace_back(x, y);
        }
    }

    void cleanup() 
    {
        frameGrabber.stop();
        cap.release();
        cv::destroyAllWindows();
    }

    Args args;
    cv::VideoCapture cap;
    FrameGrabber frameGrabber;
    cv::Mat refFrame, currentFrame;
    std::shared_ptr<XFDetector> xfeatDetector;
    std::unordered_map<std::string, torch::Tensor> refKeypointsDescriptors;
    std::string windowName = "Real-time matching - Press 's' to set the reference frame.";
    std::vector<cv::Point2f> corners = { {50.f, 50.f}, 
                                         {static_cast<float>(args.width) - 50.f, 50.f}, 
                                         {static_cast<float>(args.width) - 50.f, static_cast<float>(args.height) - 50.f}, 
                                         {50.f, static_cast<float>(args.height) - 50.f} };
    std::vector<double> timeList;
    double FPS = 0.0;
    const int maxCnt = 30;
};

int main(int argc, char** argv) 
{
    Args args;

    MatchingDemo demo(args);
    demo.main_loop();
    return 0;
}
