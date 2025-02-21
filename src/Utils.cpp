# include "Utils.h"
namespace tensorFly {
    torch::Tensor vectorToTensor(const std::vector<std::vector<size_t>> &vec) {
        // Flatten the nested vector
        std::vector<size_t> flat_vec;
        for (const auto &inner_vec: vec) {
            flat_vec.insert(flat_vec.end(), inner_vec.begin(), inner_vec.end());
        }

        // Create a tensor from the flattened vector
        auto tensor = torch::from_blob(flat_vec.data(),
                                       {static_cast<long>(vec.size()), static_cast<long>(vec[0].size())}, torch::kLong);
        return tensor.clone(); // Clone to ensure the tensor owns its data
    }
    template <typename data_type>
    torch::Tensor minMaxNormalizePerColumn(const torch::Tensor& tensor, const std::vector<data_type>& min_vals, const std::vector<data_type>& max_vals) {
        // Check that the sizes of min_vals and max_vals match the number of columns
        if (min_vals.size() != tensor.size(1) || max_vals.size() != tensor.size(1)) {
            throw std::invalid_argument("min_vals and max_vals must match the number of columns in the tensor.");
        }

        torch::Tensor normalized_tensor = tensor.clone(); // Create a copy to hold the normalized values

        for (int i = 0; i < tensor.size(1); ++i) {
            // Get the current column
            auto column = tensor.index({torch::indexing::Slice(), i});

            // Calculate the specified min and max for the current column
            float min_val = min_vals[i];
            float max_val = max_vals[i];

            // Calculate the current min and max of the column
            auto column_min = column.min().item<float>();
            auto column_max = column.max().item<float>();

            // Perform min-max normalization
            normalized_tensor.index_put_({torch::indexing::Slice(), i}, (column - column_min) / (column_max - column_min) * (max_val - min_val) + min_val);
        }

        return normalized_tensor;
    }
    torch::Tensor sample_from_categorical(const torch::Tensor& probs) {
        torch::Tensor normalized_probs = probs / probs.sum(1, /*keepdim=*/true);
        return torch::multinomial(normalized_probs, 1, /*replacement=*/true).view({-1, 1});
    }

}