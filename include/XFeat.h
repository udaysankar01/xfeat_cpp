#ifndef XFEAT_H
#define XFEAT_H

#include "XFModel.h"
#include "InterpolateSparse2d.h"
#include <opencv2/opencv.hpp>
#include <tuple>
#include <filesystem>

namespace XFeat
{   

    class XFDetector
    {
    public:
        XFDetector(int _top_k=4096, float _detection_threshold=0.05, bool use_cuda=true);
        void detectAndCompute(torch::Tensor &x, std::unordered_map<std::string, torch::Tensor> &result);
        void match(torch::Tensor &feats1, torch::Tensor &feats2, torch::Tensor &idx0, torch::Tensor &idx1, float _min_cossim=-1.0);
        void match_xfeat(cv::Mat &img1, cv::Mat &img2, cv::Mat &mkpts_0, cv::Mat &mkpts_1);
        torch::Tensor parseInput(const cv::Mat &img);
        std::tuple<torch::Tensor, double, double> preprocessTensor(torch::Tensor &x);
        cv::Mat tensorToMat(const torch::Tensor &tensor);

        cv::Mat tensorTo1DMat(const torch::Tensor& tensor);
        std::unordered_map<std::string,at::Tensor> convertToTorch(const cv::Mat &descriptors, const std::vector<cv::KeyPoint> &keypoints);
        torch::Tensor cvToTorch(const cv::Mat& descriptors);
        torch::Tensor cvToTorch(const std::vector<cv::KeyPoint>& keypoints);
        
        void extractFeatures(const cv::Mat& img,std::vector<cv::KeyPoint>& kps,cv::Mat& decs);
        void matchFeatures(const cv::Mat& desc1,const cv::Mat& desc2, std::vector<cv::DMatch>& good_matches);

    private: 
        torch::Tensor getKptsHeatmap(torch::Tensor &kpts, float softmax_temp=1.0);
        torch::Tensor NMS(torch::Tensor &x, float threshold = 0.05, int kernel_size = 5);
        std::string getWeightsPath(std::string weights);
        
        std::string weights;
        int top_k;
        float min_cossim;
        float detection_threshold;
        torch::DeviceType device_type;
        std::shared_ptr<XFeatModel> model;
        std::shared_ptr<InterpolateSparse2d> bilinear, nearest;
    };

} // namespace XFeat


#endif // XFEAT_H