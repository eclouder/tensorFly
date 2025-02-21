#pragma once
#include "include.h"
#include <torch/torch.h>
#include "RLEngine/collection.h"
namespace tensorFly{
namespace RLEngine{
    class network{
//        virtual void train(torch::Tensor input);
    };
    class PolicyNet : public torch::nn::Module {
    public:
        PolicyNet(int64_t state_dim, int64_t hidden_dim, int64_t action_dim) {
            fc1 = register_module("fc1", torch::nn::Linear(state_dim, hidden_dim));
            fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, action_dim));
        }

        torch::Tensor forward(torch::Tensor x) {
            x = torch::relu(fc1->forward(x));
            return torch::softmax(fc2->forward(x), /*dim=*/1);
        }

    private:
        torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    };

    class ValueNet : public torch::nn::Module {
    public:
        ValueNet(int64_t state_dim, int64_t hidden_dim) {
            fc1 = register_module("fc1", torch::nn::Linear(state_dim, hidden_dim));
            fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, 1));
        }

        torch::Tensor forward(torch::Tensor x) {
            x = torch::relu(fc1->forward(x));
            return fc2->forward(x);
        }

    private:
        torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    };

    class torchNet : public network { // Assuming 'network' is defined elsewhere
        friend class env;
    public:
        enum netType {
            mlp,
            // unSetting // Uncomment if needed
        };

        torchNet(size_t action_size, size_t space_size)
                : actions_size(action_size), spaces_size(space_size),
                  action_module(space_size, 128, action_size), // Correct initialization
                  value_module(space_size, 128) { // Correct initialization

            action_optimizer = std::make_shared<torch::optim::Adam>(
                    action_module.parameters(), /*lr=*/0.001);
            value_optimizer = std::make_shared<torch::optim::Adam>(
                    value_module.parameters(), /*lr=*/0.001);
        }

//        void train(torch::Tensor input) override;
        torch::nn::Module get_action_module(netType net_type = mlp) {
            // Implement your logic here
        }

        torch::nn::Module get_value_module(netType net_type = mlp) {
            // Implement your logic here
        }

    private:
        PolicyNet action_module;   // Use member initialization in the constructor
        ValueNet value_module;     // Use member initialization in the constructor
        size_t actions_size;
        size_t spaces_size;
        std::vector<size_t> actions_choice_size = {3, 3};
        std::vector<std::vector<size_t>> actions;
        std::vector<std::vector<size_t>> spaces;
        std::shared_ptr<torch::optim::Adam> action_optimizer;
        std::shared_ptr<torch::optim::Adam> value_optimizer;

    };
}
}