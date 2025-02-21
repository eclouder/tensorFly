#include "RLEngine/net.h"
using namespace tensorFly;
using namespace RLEngine;

class Module: public torch::nn::Module{
public:
    virtual torch::Tensor forward(torch::Tensor inputs) {};
};
//void network::train(torch::Tensor input) {}
class RLmodule : public torch::nn::Module {
public:
    RLmodule(const std::vector<std::vector<size_t>>& inputVec) {
        for (const auto& innerVec : inputVec) {
            size_t inputSize = innerVec.size();
            auto linearLayer = register_module("linear_" + std::to_string(linearLayers.size()),
                                               torch::nn::Linear(inputSize, 1024));
            linearLayers.push_back(linearLayer);
        }
    }

    torch::Tensor forward(const torch::Tensor inputs) {
        for(auto &layer:linearLayers){

        }
//        return torch::cat(outputs, 0);
    }

private:
    std::vector<torch::nn::Linear> linearLayers;
};
struct MLP : torch::nn::Module {
    MLP(int64_t input_size, int64_t hidden_size, int64_t output_size)
            : fc1(register_module("fc1", torch::nn::Linear(input_size, hidden_size))),
              fc2(register_module("fc2", torch::nn::Linear(hidden_size, output_size))) {}

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1(x));
        x = fc2(x);
        return x;
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};


struct PolicyNetContinuous : torch::nn::Module {
    PolicyNetContinuous(int64_t state_dim, int64_t hidden_dim, int64_t action_dim) {
        fc1 = register_module("fc1", torch::nn::Linear(state_dim, hidden_dim));
        fc_mu = register_module("fc_mu", torch::nn::Linear(hidden_dim, action_dim));
        fc_std = register_module("fc_std", torch::nn::Linear(hidden_dim, action_dim));
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        torch::Tensor mu = 2.0 * torch::tanh(fc_mu->forward(x));
        torch::Tensor std = torch::softplus(fc_std->forward(x));
        return std::make_tuple(mu, std);
    }

    torch::nn::Linear fc1{nullptr}, fc_mu{nullptr}, fc_std{nullptr};
};



