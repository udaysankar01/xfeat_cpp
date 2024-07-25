#include "XFeat.h"

namespace XFeat
{
    
    // XFDetector Implementation
    XFDetector::XFDetector(std::shared_ptr<XFeatModel> _model) : model(_model)
    {
        interpolator = InterpolateSparse2d("bilinear", false);      
        detection_threshold = 0.05;
    }

    std::vector<std::unordered_map<std::string, torch::Tensor>> XFDetector::detectAndCompute(torch::Tensor x, int top_k, bool cuda)
    {
        /*
            Compute sparse keypoints & descriptors. Supports batched mode.

            input:
                x -> torch.Tensor(B, C, H, W): grayscale or rgb image
                top_k -> int: keep best k features
            return:
                List[Dict]: 
                    'keypoints'    ->   torch.Tensor(N, 2): keypoints (x,y)
                    'scores'       ->   torch.Tensor(N,): keypoint scores
                    'descriptors'  ->   torch.Tensor(N, 64): local features
        */

        bool use_cuda = cuda && torch::cuda::is_available();
        torch::DeviceType device_type = (use_cuda) ? torch::kCUDA : torch::kCPU;
        torch::Device device(device_type);

        float rh1, rw1;
        std::tie(x, rh1, rw1) = preprocessTensor(x);

        auto   B = x.size(0);
        auto _H1 = x.size(2);
        auto _W1 = x.size(3);

        // forward pass
        auto out = model->forward(x);
        torch::Tensor M1, K1, H1;
        std::tie(M1, K1, H1) = out;
        M1 = torch::nn::functional::normalize(M1, torch::nn::functional::NormalizeFuncOptions().dim(1));

        // convert logits to heatmap and extract keypoints
        torch::Tensor K1h = getKptsHeatmap(K1);
        torch::Tensor mkpts = NMS(K1h, detection_threshold, 5);

        // compute reliability scores
        InterpolateSparse2d _nearest  = InterpolateSparse2d("nearest");
        InterpolateSparse2d _bilinear = InterpolateSparse2d("bilinear");
        auto scores = (_nearest.forward(K1h, mkpts, _H1, _W1) * _bilinear.forward(H1, mkpts, _H1, _W1)).squeeze(-1);
        auto mask = torch::all(mkpts == 0, -1);
        scores.masked_fill_(mask, -1);

        // Select top-k features
        torch::Tensor idxs = scores.neg().argsort(-1, false);
        auto mkpts_x = mkpts.index({torch::indexing::Ellipsis, 0})
                            .gather(-1, idxs)
                            .index({torch::indexing::Slice(), torch::indexing::Slice(0, top_k)});
        auto mkpts_y = mkpts.index({torch::indexing::Ellipsis, 1})
                            .gather(-1, idxs)
                            .index({torch::indexing::Slice(), torch::indexing::Slice(0, top_k)});
        mkpts_x = mkpts_x.unsqueeze(-1);
        mkpts_y = mkpts_y.unsqueeze(-1);
        mkpts = torch::cat({mkpts_x, mkpts_y}, -1);
        scores = scores.gather(-1, idxs).index({torch::indexing::Slice(), torch::indexing::Slice(0, top_k)});

        // Interpolate descriptors at kpts positions
        torch::Tensor feats = interpolator.forward(M1, mkpts, _H1, _W1);

        // L2-Normalize
        feats = torch::nn::functional::normalize(feats, torch::nn::functional::NormalizeFuncOptions().dim(-1));

        auto min_val = feats.min();
        auto max_val = feats.max();

        // correct kpt scale
        torch::Tensor scaling_factors = torch::tensor({rw1, rh1}, mkpts.options()).view({1, 1, -1});
        mkpts = mkpts * scaling_factors;

        std::vector<std::unordered_map<std::string, torch::Tensor>> result;
        for (int b = 0; b < B; b++) {
            auto valid = scores[b] > 0;
            std::unordered_map<std::string, torch::Tensor> item;
            item["keypoints"] = mkpts[b].index({valid});
            item["scores"] = scores[b].index({valid});
            item["descriptors"] = feats[b].index({valid});
            result.push_back(item);
        }
        return result;
    }   

    std::tuple<torch::Tensor, torch::Tensor> XFDetector::match(torch::Tensor feats1, torch::Tensor feats2, float min_cossim)
    {   
        // compute cossine similarity between feats1 and feats2
        torch::Tensor cossim = torch::matmul(feats1, feats2.t());
        torch::Tensor cossim_t = torch::matmul(feats2, feats1.t());

        torch::Tensor match12, match21;
        std::tie(std::ignore, match12) = cossim.max(1);
        std::tie(std::ignore, match21) = cossim_t.max(1);

        // index tensor
        torch::Tensor idx0 = torch::arange(match12.size(0), match12.options());
        torch::Tensor mutual = match21.index({match12}) == idx0;

        torch::Tensor idx1;
        if (min_cossim > 0)
        {
            std::tie(cossim, std::ignore) = cossim.max(1);
            torch::Tensor good = cossim > min_cossim;
            idx0 = idx0.index({mutual & good});
            idx1 = match12.index({mutual & good});
        }
        else
        {
            idx0 = idx0.index({mutual});
            idx1 = match12.index({mutual}); 
        }

        return std::make_tuple(idx0, idx1);
    }

    std::pair<cv::Mat, cv::Mat> XFDetector::match_xfeat(cv::Mat& img1, cv::Mat& img2, int top_k, float min_cossim)
    {   
        torch::Tensor tensor_img1 = parseInput(img1);
        torch::Tensor tensor_img2 = parseInput(img2);

        auto out1 = detectAndCompute(tensor_img1, top_k, /*use_cuda*/true)[0]; // no batches
        auto out2 = detectAndCompute(tensor_img2, top_k, /*use_cuda*/true)[0];

        torch::Tensor idxs0, idxs1;
        std::tie(idxs0, idxs1) = match(out1["descriptors"], out2["descriptors"], min_cossim);

        torch::Tensor mkpts_0 = out1["keypoints"].index({idxs0});
        torch::Tensor mkpts_1 = out2["keypoints"].index({idxs1});
        cv::Mat mkpts_0_cv = tensorToMat(mkpts_0);
        cv::Mat mkpts_1_cv = tensorToMat(mkpts_1);

        return std::make_pair(mkpts_0_cv, mkpts_1_cv);
    }

    void XFDetector::warp_corners_and_draw_matches(cv::Mat& ref_points, cv::Mat& dst_points, cv::Mat& img1, cv::Mat& img2)
    {      
        // Check if there are enough points to find a homography
        if (ref_points.rows < 4 || dst_points.rows < 4) {
            std::cerr << "Not enough points to compute homography" << std::endl;
            return;
        }

        cv::Mat mask;
        cv::Mat H = cv::findHomography(ref_points, dst_points, cv::USAC_MAGSAC, 10.0, mask, 1000, 0.994);
        if (H.empty()) {
            std::cerr << "Homography matrix is empty" << std::endl;
            return;
        }
        mask = mask.reshape(1);

        float h = img1.rows;
        float w = img1.cols;
        std::vector<cv::Point2f> corners_img1 = {cv::Point2f(    0,     0), 
                                                 cv::Point2f(w - 1,     0), 
                                                 cv::Point2f(w - 1, h - 1), 
                                                 cv::Point2f(    0, h - 1)};
        std::vector<cv::Point2f> warped_corners;
        cv::perspectiveTransform(corners_img1, warped_corners, H);

        cv::Mat img2_with_corners = img2.clone();
        for (size_t i = 0; i < warped_corners.size(); ++i) {
            cv::line(img2_with_corners, warped_corners[i], warped_corners[(i+1) % warped_corners.size()], cv::Scalar(0, 255, 0), 4);
        }

        // prepare keypoints and matches for drawMatches function
        std::vector<cv::KeyPoint> keypoints1, keypoints2;
        std::vector<cv::DMatch> matches;
        for (int i = 0; i < mask.rows; ++i) {
            keypoints1.emplace_back(ref_points.at<cv::Point2f>(i, 0), 5);
            keypoints2.emplace_back(dst_points.at<cv::Point2f>(i, 0), 5);
            if (mask.at<uchar>(i, 0))
                matches.emplace_back(i, i, 0);
        }
        
        // Draw inlier matches
        cv::Mat img_matches;
        if (!keypoints1.empty() && !keypoints2.empty() && !matches.empty()) {
            cv::drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            cv::imshow("Matches", img_matches);
            cv::waitKey(0); // Wait for a key press
        } else {
            std::cerr << "Keypoints or matches are empty, cannot draw matches" << std::endl;
        }
    }

    torch::Tensor XFDetector::parseInput(cv::Mat &img)
    {   
        // if the image is grayscale
        if (img.channels() == 1)
        {
            torch::Tensor tensor = torch::from_blob(img.data, {1, img.rows, img.cols, 1}, torch::kByte);
            tensor = tensor.permute({0, 3, 1, 2}).to(torch::kFloat) / 255.0;
            return tensor;
        }

        // if image is in RGB format
        if (img.channels() == 3) {
            torch::Tensor tensor = torch::from_blob(img.data, {1, img.rows, img.cols, 3}, torch::kByte);
            tensor = tensor.permute({0, 3, 1, 2}).to(torch::kFloat) / 255.0;
            return tensor;
        }

        // If the image has an unsupported number of channels, throw an error
        throw std::invalid_argument("Unsupported number of channels in the input image.");  
    }

    std::tuple<torch::Tensor, double, double> XFDetector::preprocessTensor(torch::Tensor x)
    {
        /* 
            Guarentees that image is divisible by 32 to avoid aliasing artifacts.
        */

        // ensure the tensor has the correct type
        x = x.to(torch::kFloat);

        // calculate new size divisible by 32
        int H = x.size(-2);
        int W = x.size(-1);
        int64_t _H = (H / 32) * 32;
        int64_t _W = (W / 32) * 32;

        // calculate resize ratios
        double rh = static_cast<double>(H) / _H;
        double rw = static_cast<double>(W) / _W;

        std::vector<int64_t> size_array = {_H, _W};
        x = torch::nn::functional::interpolate(x, torch::nn::functional::InterpolateFuncOptions().size(size_array)
                                                                                             .mode(torch::kBilinear)
                                                                                          .align_corners(false));
        return std::make_tuple(x, rh, rw);
    }

    torch::Tensor XFDetector::getKptsHeatmap(torch::Tensor kpts, float softmax_temp)
    {   
        torch::Tensor scores = torch::nn::functional::softmax(kpts * softmax_temp, torch::nn::functional::SoftmaxFuncOptions(1));
        scores = scores.index({torch::indexing::Slice(), torch::indexing::Slice(0, 64), torch::indexing::Slice(), torch::indexing::Slice()});

        int B = scores.size(0);
        int H = scores.size(2);
        int W = scores.size(3);

        // reshape and permute the tensor to form heatmap
        torch::Tensor heatmap = scores.permute({0, 2, 3, 1}).reshape({B, H, W, 8, 8});
        heatmap = heatmap.permute({0, 1, 3, 2, 4}).reshape({B, 1, H*8, W*8});
        return heatmap;
    }

    torch::Tensor XFDetector::NMS(torch::Tensor x, float threshold, int kernel_size)
    {
        int B = x.size(0);
        int H = x.size(2);
        int W = x.size(3);
        int pad = kernel_size / 2;

        auto local_max = torch::nn::functional::max_pool2d(x, torch::nn::functional::MaxPool2dFuncOptions(kernel_size).stride(1)
                                                                                                                      .padding(pad));
        auto pos = (x == local_max) & (x > threshold);
        std::vector<torch::Tensor> pos_batched;
        for (int b = 0; b < pos.size(0); ++b) 
        {
            auto k = pos[b].nonzero();
            k = k.index({torch::indexing::Ellipsis, torch::indexing::Slice(1, torch::indexing::None)}).flip(-1);
            pos_batched.push_back(k);
        }

        int pad_val = 0;
        for (const auto& p : pos_batched) {
            pad_val = std::max(pad_val, static_cast<int>(p.size(0)));
        }
        
        torch::Tensor pos_tensor = torch::zeros({B, pad_val, 2}, torch::TensorOptions().dtype(torch::kLong).device(x.device()));
        for (int b = 0; b < B; ++b) {
            if (pos_batched[b].size(0) > 0) {
                pos_tensor[b].narrow(0, 0, pos_batched[b].size(0)) = pos_batched[b];
            }
        }

        return pos_tensor;
    }

    cv::Mat XFDetector::tensorToMat(const torch::Tensor &tensor)
    {
        // ensure tesnor is on CPU and convert to float
        torch::Tensor cpu_tensor = tensor.to(torch::kCPU).to(torch::kFloat);
        cv::Mat mat(cpu_tensor.size(0), 2, CV_32F);
        std::memcpy(mat.data, cpu_tensor.data_ptr<float>(), cpu_tensor.numel() * sizeof(float));
        return mat;
    }


} // namespace XFeat 