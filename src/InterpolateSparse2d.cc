#include "InterpolateSparse2d.h"

namespace XFeat
{
    InterpolateSparse2d::InterpolateSparse2d(const std::string& mode, bool align_corners)
    : mode(mode), align_corners(align_corners)
    {
    }

    torch::Tensor InterpolateSparse2d::normgrid(torch::Tensor x, int H, int W)
    {
        // normalize coordinates to [-1, 1]
        torch::Tensor size_tensor = torch::tensor({W - 1, H - 1}, x.options());
        return 2.0 * (x / size_tensor) - 1.0;
    }

    torch::Tensor InterpolateSparse2d::forward(torch::Tensor x, torch::Tensor pos, int H, int W)
    {
        // normalize the positions
        torch::Tensor grid = normgrid(pos, H, W).unsqueeze(-2).to(x.dtype());

        // grid sampling  ---- EDIT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if (mode == "bilinear")
        {
            x = torch::nn::functional::grid_sample(x, grid, torch::nn::functional::GridSampleFuncOptions().mode(torch::kBilinear).align_corners(align_corners));
        }
        else if (mode == "nearest")
        {
            x = torch::nn::functional::grid_sample(x, grid, torch::nn::functional::GridSampleFuncOptions().mode(torch::kNearest).align_corners(align_corners));
        }   
        else
        {
            std::cerr << "Choose either 'bilinear' or 'nearest'." << std::endl;
            exit(EXIT_FAILURE);
        }

        //reshape output to [B, N, C]
        return x.permute({0, 2, 3, 1}).squeeze(-2);
    }
}