#pragma once
# include "include.h"
namespace tensorFly{
    namespace RLEngine{
        struct Collection{
            std::vector<torch::Tensor>states;
            std::vector<torch::Tensor> actions;
            std::vector<torch::Tensor> next_states;
            std::vector<torch::Tensor> reward;
            std::vector<torch::Tensor> done;
        };
    }
}