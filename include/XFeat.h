#ifndef XFEAT_H
#define XFEAT_H

#include "XFModel.h"
#include "InterpolateSparse2d.h"
#include <opencv2/opencv.hpp>
#include <tuple>

namespace XFeat
{

    class XFDetector
    {
    public:
        XFDetector(std::shared_ptr<XFeatModel> _model);
        std::vector<std::unordered_map<std::string, torch::Tensor>> detectAndCompute(torch::Tensor x, int top_k, bool cuda);
        std::tuple<torch::Tensor, torch::Tensor> match(torch::Tensor feats1, torch::Tensor feats2, float min_cossim = -1);
        std::pair<cv::Mat, cv::Mat> match_xfeat(cv::Mat& img1, cv::Mat& img2, int top_k, float min_cossim = -1);
        void warp_corners_and_draw_matches(cv::Mat& mkpts_0, cv::Mat& mkpts_1, cv::Mat& img1, cv::Mat& img2);
        
    private:
        torch::Tensor parseInput(cv::Mat &img);
        torch::Tensor getKptsHeatmap(torch::Tensor kpts, float softmax_temp=1.0);
        std::tuple<torch::Tensor, double, double> preprocessTensor(torch::Tensor x);
        torch::Tensor NMS(torch::Tensor x, float threshold = 0.05, int kernel_size = 5);
        cv::Mat tensorToMat(const torch::Tensor &tensor);
        
        std::shared_ptr<XFeatModel> model;
        InterpolateSparse2d interpolator;
        float detection_threshold;
    };

} // namespace XFeat


#endif // XFEAT_H