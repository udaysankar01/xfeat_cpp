#include "XFeat.h"

namespace XFeat
{
    
    // XFDetector Implementation
    XFDetector::XFDetector(int _top_k, float _detection_threshold, bool use_cuda) 
        : top_k(_top_k), 
          detection_threshold(_detection_threshold)
    {   
        // load the model
        weights = "weights/xfeat.pt";
        model = std::make_shared<XFeatModel>();
        torch::serialize::InputArchive archive;
        archive.load_from(getWeightsPath(weights));
        model->load(archive);
        std::cout << "Model weights loaded successfully." << std::endl;

        // move the model to device
        device_type = (use_cuda && torch::cuda::is_available()) ? torch::kCUDA : torch::kCPU;
        torch::Device device(device_type);
        std::cout << "Device: " << device << '\n';
        model->to(device);

        // load the bilinear interpolator
        bilinear = std::make_shared<InterpolateSparse2d>("bilinear");     
        nearest  = std::make_shared<InterpolateSparse2d>("nearest"); 

        // cosine similarity matching threshold
        min_cossim = -1;
    }

    void XFDetector::detectAndCompute(torch::Tensor& x, std::unordered_map<std::string, torch::Tensor> &result)
    {   
        torch::Device device(device_type);
        x = x.to(device);

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
        auto scores = (nearest->forward(K1h, mkpts, _H1, _W1) * bilinear->forward(H1, mkpts, _H1, _W1)).squeeze(-1);
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
        torch::Tensor feats = bilinear->forward(M1, mkpts, _H1, _W1);

        // L2-Normalize
        feats = torch::nn::functional::normalize(feats, torch::nn::functional::NormalizeFuncOptions().dim(-1));

        // correct kpt scale
        torch::Tensor scaling_factors = torch::tensor({rw1, rh1}, mkpts.options()).view({1, 1, -1});
        mkpts = mkpts * scaling_factors;

        auto valid = scores[0] > 0;
        result["keypoints"] = mkpts[0].index({valid});
        result["scores"] = scores[0].index({valid});
        result["descriptors"] = feats[0].index({valid});
    }   

    void XFDetector::match(torch::Tensor& feats1, torch::Tensor& feats2, torch::Tensor &idx0, torch::Tensor &idx1, float _min_cossim)
    {   
        // set the min_cossim
        min_cossim = _min_cossim;

        // compute cossine similarity between feats1 and feats2
        torch::Tensor cossim = torch::matmul(feats1, feats2.t());
        torch::Tensor cossim_t = torch::matmul(feats2, feats1.t());

        torch::Tensor match12, match21;
        std::tie(std::ignore, match12) = cossim.max(1);
        std::tie(std::ignore, match21) = cossim_t.max(1);

        // index tensor
        idx0 = torch::arange(match12.size(0), match12.options());
        torch::Tensor mutual = match21.index({match12}) == idx0;

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
    }

    void XFDetector::match_xfeat(cv::Mat& img1, cv::Mat& img2, cv::Mat &mkpts_0, cv::Mat &mkpts_1)
    {   
        torch::Tensor tensor_img1 = parseInput(img1);
        torch::Tensor tensor_img2 = parseInput(img2);

        std::unordered_map<std::string, at::Tensor> out1, out2;
        detectAndCompute(tensor_img1, out1); // no batches
        detectAndCompute(tensor_img2, out2);

        torch::Tensor idxs0, idxs1;
        match(out1["descriptors"], out2["descriptors"], idxs0, idxs1, -1.0);

        torch::Tensor mkpts_0_tensor = out1["keypoints"].index({idxs0});
        torch::Tensor mkpts_1_tensor = out2["keypoints"].index({idxs1});
        mkpts_0 = tensorToMat(mkpts_0_tensor);
        mkpts_1 = tensorToMat(mkpts_1_tensor);
    }

    torch::Tensor XFDetector::parseInput(const cv::Mat &img)
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

    std::tuple<torch::Tensor, double, double> XFDetector::preprocessTensor(torch::Tensor& x)
    {
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

    torch::Tensor XFDetector::getKptsHeatmap(torch::Tensor& kpts, float softmax_temp)
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

    torch::Tensor XFDetector::NMS(torch::Tensor& x, float threshold, int kernel_size)
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

    cv::Mat XFDetector::tensorToMat(const torch::Tensor& tensor)
    {   
        // ensure tesnor is on CPU and convert to float
        torch::Tensor cpu_tensor = tensor.to(torch::kCPU).to(torch::kFloat);
        cv::Mat mat(cpu_tensor.size(0), cpu_tensor.size(1), CV_32F);
        std::memcpy(mat.data, cpu_tensor.data_ptr<float>(), cpu_tensor.numel() * sizeof(float));
        return mat;
    }

    std::string XFDetector::getWeightsPath(std::string weights)
    {
        std::filesystem::path current_file = __FILE__;
        std::filesystem::path parent_dir = current_file.parent_path();
        std::filesystem::path full_path = parent_dir / ".." / weights;
        full_path = std::filesystem::absolute(full_path);

        return static_cast<std::string>(full_path);   
    }

    cv::Mat XFDetector::tensorTo1DMat(const torch::Tensor& tensor)
    {   
        // ensure tesnor is on CPU and convert to float
        torch::Tensor cpu_tensor = tensor.to(torch::kCPU).to(torch::kFloat);
        // cv::Mat mat(cpu_tensor.size(0),1, CV_32F);
        cv::Mat mat(cpu_tensor.size(0),1, CV_32F);
        std::memcpy(mat.data, cpu_tensor.data_ptr<float>(), cpu_tensor.numel() * sizeof(float));
        return mat;
    }

    std::unordered_map<std::string, at::Tensor> XFDetector::convertToTorch(const cv::Mat &descriptors, const std::vector<cv::KeyPoint> &keypoints)
    {
        std::unordered_map<std::string, at::Tensor> output;
        output["keypoints"] = cvToTorch(descriptors);
        output["descriptors"] = cvToTorch(keypoints);
    
        return output;
    }

    torch::Tensor XFDetector::cvToTorch(const cv::Mat& descriptors)
    {
        torch::Tensor desc_tensor = torch::from_blob(descriptors.data, {descriptors.rows, descriptors.cols}, torch::kFloat);
        desc_tensor = desc_tensor.clone(); 

        return desc_tensor;

    }

    torch::Tensor XFDetector::cvToTorch(const std::vector<cv::KeyPoint>& keypoints)
    {
        std::vector<float> kpts_vec;
        for (const auto &kp : keypoints)
        {
            kpts_vec.push_back(kp.pt.x);
            kpts_vec.push_back(kp.pt.y);
        }
        torch::Tensor kpts_tensor = torch::from_blob(kpts_vec.data(), {static_cast<long>(keypoints.size()), 2}, torch::kFloat).clone();

        return kpts_tensor;
    }

    void XFDetector::extractFeatures(const cv::Mat& img,std::vector<cv::KeyPoint>& keypoints,cv::Mat& descs)
    {
        torch::Tensor tensor_img = this->parseInput(img);
        std::unordered_map<std::string,at::Tensor> out;
        this->detectAndCompute(tensor_img,out);
        cv::Mat kps = this->tensorToMat(out["keypoints"]);
        descs = this->tensorToMat(out["descriptors"]);
    
        keypoints.clear();
    
        for(int i = 0; i <  kps.rows; i++)
        {
            cv::KeyPoint kp(kps.at<float>(i,0),kps.at<float>(i,1),0);
            keypoints.push_back(kp);
        }
    }

    void XFDetector::matchFeatures(const cv::Mat& desc1,const cv::Mat& desc2, std::vector<cv::DMatch>& good_matches)
    {
        torch::Tensor out_tensor_1 = this->cvToTorch(desc1);
        torch::Tensor out_tensor_2 = this->cvToTorch(desc2);
    
        torch::Tensor idx0,idx1;
        this->match(out_tensor_1,out_tensor_2,idx0,idx1,0.82);
    
        cv::Mat index0 = this->tensorTo1DMat(idx0);
        cv::Mat index1 = this->tensorTo1DMat(idx1);
    
        good_matches.clear();
    
        for(int i = 0; i < index0.rows; i++)
        {
            cv::DMatch match(index0.at<float>(i),index1.at<float>(i),0);
            good_matches.push_back(match);
        }
    }


} // namespace XFeat 