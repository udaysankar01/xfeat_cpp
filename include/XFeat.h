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
        std::vector<std::unordered_map<std::string, torch::Tensor>> detectAndCompute(torch::Tensor& x);
        std::tuple<torch::Tensor, torch::Tensor> match(torch::Tensor& feats1, torch::Tensor& feats2, float _min_cossim=-1.0);
        std::pair<cv::Mat, cv::Mat> match_xfeat(cv::Mat& img1, cv::Mat& img2);
        torch::Tensor parseInput(cv::Mat& img);
        std::tuple<torch::Tensor, double, double> preprocessTensor(torch::Tensor& x);
        cv::Mat tensorToMat(const torch::Tensor& tensor);
        
    private:
        torch::Tensor getKptsHeatmap(torch::Tensor& kpts, float softmax_temp=1.0);
        torch::Tensor NMS(torch::Tensor& x, float threshold = 0.05, int kernel_size = 5);
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