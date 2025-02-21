#pragma  once
#include "include.h"
#include "torch/torch.h"
namespace tensorFly {
//    double normalize(double value, double min, double max) {
//        return (value - min) / (max - min);
//    }
    torch::Tensor vectorToTensor(const std::vector<std::vector<size_t>> &vec);
    template <typename data_type>
    torch::Tensor minMaxNormalizePerColumn(const torch::Tensor& tensor, const std::vector<data_type>& min_vals, const std::vector<data_type>& max_vals);
    torch::Tensor sample_from_categorical(const torch::Tensor& probs);


}
