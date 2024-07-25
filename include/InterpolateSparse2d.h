#ifndef INTERPOLATE_SPARSE_2D_H
#define INTERPOLATE_SPARSE_2D_H

#include <torch/torch.h>

namespace XFeat
{
    class InterpolateSparse2d : public torch::nn::Module
    {
    public:
        InterpolateSparse2d(const std::string& mode = "bilinear", bool align_corners = false);
        torch::Tensor forward(torch::Tensor x, torch::Tensor pos, int H, int W);

    private:
        torch::Tensor normgrid(torch::Tensor x, int H, int W);

        std::string mode;
        bool align_corners;
    };
}

#endif // INTERPOLATE_SPARSE_2D_H