#include "XFModel.h"

namespace XFeat
{
    /////////////////////////////  The Basic Layer  ////////////////////////////////////
    BasicLayerImpl::BasicLayerImpl(int in_channels, 
                    int out_channels, 
                    int kernel_size,
                    int stride,
                    int padding)
    {
        layer = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                .padding(padding)
                .stride(stride)
                .dilation(1)
                .bias(false)),
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels).affine(false)),
            torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))
        );
        register_module("layer", layer);
    }

    torch::Tensor BasicLayerImpl::forward(torch::Tensor x) 
    {
        return layer->forward(x);
    }


    //////////////////////////////// The XFeat Model  ///////////////////////////////
    XFeatModel::XFeatModel()
    {
        norm = torch::nn::InstanceNorm2d(1);

        // CNN Backbone and Heads

        skip1 = torch::nn::Sequential(
            torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(4).stride(4)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 24, 1).stride(1).padding(0))
        );
        
        block1 = torch::nn::Sequential(
            BasicLayer(1,  4, 3, 1, 1),
            BasicLayer(4,  8, 3, 2, 1),
            BasicLayer(8,  8, 3, 1, 1),
            BasicLayer(8, 24, 3, 2, 1)
        );

        block2 = torch::nn::Sequential(
            BasicLayer(24, 24, 3, 1, 1),
            BasicLayer(24, 24, 3, 1, 1)
        );

        block3 = torch::nn::Sequential(
            BasicLayer(24, 64, 3, 2, 1),
            BasicLayer(64, 64, 3, 1, 1),
            BasicLayer(64, 64, 1, 1, 0)
        );

        block4 = torch::nn::Sequential(
            BasicLayer(64, 64, 3, 2, 1),
            BasicLayer(64, 64, 3, 1, 1),
            BasicLayer(64, 64, 3, 1, 1)
        );

        block5 = torch::nn::Sequential(
            BasicLayer( 64, 128, 3, 2, 1),
            BasicLayer(128, 128, 3, 1, 1),
            BasicLayer(128, 128, 3, 1, 1),
            BasicLayer(128,  64, 1, 1, 0)
        );

        block_fusion = torch::nn::Sequential(
            BasicLayer(64, 64, 3, 1, 1),
            BasicLayer(64, 64, 3, 1, 1),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 1).padding(0))
        );

        heatmap_head = torch::nn::Sequential(
            BasicLayer(64, 64, 1, 1, 0),
            BasicLayer(64, 64, 1, 1, 0),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 1, 1)),
            torch::nn::Sigmoid()
        );

        keypoint_head = torch::nn::Sequential(
            BasicLayer(64, 64, 1, 1, 0),
            BasicLayer(64, 64, 1, 1, 0),
            BasicLayer(64, 64, 1, 1, 0),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 65, 1))
        );

        // Fine Matcher MLP

        fine_matcher = torch::nn::Sequential(
            torch::nn::Linear(128, 512),
            torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(512).affine(false)),
            torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
            torch::nn::Linear(512, 512),
            torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(512).affine(false)),
            torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
            torch::nn::Linear(512, 512),
            torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(512).affine(false)),
            torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
            torch::nn::Linear(512, 512),
            torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(512).affine(false)),
            torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
            torch::nn::Linear(512, 64)
        );


        register_module("norm", norm);
        register_module("skip1", skip1);
        register_module("block1", block1);
        register_module("block2", block2);
        register_module("block3", block3);
        register_module("block4", block4);
        register_module("block5", block5);
        register_module("block_fusion", block_fusion);
        register_module("heatmap_head", heatmap_head);
        register_module("keypoint_head", keypoint_head);
        register_module("fine_matcher", fine_matcher);
    }

    torch::Tensor XFeatModel::unfold2d(torch::Tensor x, int ws)
    {   
        /*
           Unfolds tensor in 2D with desired ws (window size) and concat the channels
        */
        auto shape = x.sizes();
        int B = shape[0], C = shape[1], H = shape[2], W = shape[3];
        x = x.unfold(2, ws, ws).unfold(3, ws, ws).reshape({B, C, H/ws, W/ws, ws*ws});
        return x.permute({0, 1, 4, 2, 3}).reshape({B, -1, H/ws, W/ws});
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> XFeatModel::forward(torch::Tensor x) 
    {   
        /* 
            input:
                x -> torch.Tensor(B, C, H, W) grayscale or rgb images
            return:
                feats      -> torch.Tensor(B, 64, H/8, W/8) dense local features
                keypoints  -> torch.Tensor(B, 65, H/8, W/8) keypoint logit map
                heatmap    -> torch.Tensor(B,  1, H/8, W/8) reliability map
        */

        // don't backprop through normalization
        torch::NoGradGuard no_grad;
        x = x.mean(1, true);
        x = norm->forward(x);

        // main backbone
        torch::Tensor x1 = block1->forward(x);
        torch::Tensor x2 = block2->forward(x1 + skip1->forward(x));
        torch::Tensor x3 = block3->forward(x2);
        torch::Tensor x4 = block4->forward(x3);
        torch::Tensor x5 = block5->forward(x4);

        // pyramid fusion
        std::vector<int64_t> size_array = {x3.size(2), x3.size(3)};
        x4 = torch::nn::functional::interpolate(x4, torch::nn::functional::InterpolateFuncOptions().size(size_array)
                                                                                                .mode(torch::kBilinear)
                                                                                                .align_corners(false));
        x5 = torch::nn::functional::interpolate(x5, torch::nn::functional::InterpolateFuncOptions().size(size_array)
                                                                                                .mode(torch::kBilinear)
                                                                                                .align_corners(false));
        torch::Tensor feats = block_fusion->forward(x3 + x4 + x5);

        // heads
        torch::Tensor heatmap = heatmap_head->forward(feats);
        torch::Tensor keypoints = keypoint_head->forward(unfold2d(x, 8));

        return std::make_tuple(feats, keypoints, heatmap);
    }
}