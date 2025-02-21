#pragma once
#include "include.h"
#include "RLEngine/net.h"
#include "torch/torch.h"
#include "RLEngine/collection.h"

namespace tensorFly{
namespace RLEngine{
//template <typename dataT>
class env{
public:

//    env(const std::vector<std::vector<dataT>>& init_actions,
//        const std::vector<std::vector<dataT>>& init_spaces,
//        torchNet net)
//            : actions(init_actions), spaces(init_spaces), net(net){}
//    env(const std::vector<std::vector<dataT>>& init_actions,
//        const std::vector<std::vector<dataT>>& init_spaces,
//        const std::vector<std::string>& init_action_names,
//        const std::vector<std::string>& init_spaces_names,
//        const std::vector<std::vector<std::string>>& init_actions_describe,
//        const std::vector<std::vector<std::string>>& init_spaces_describe)
//            : actions(init_actions), spaces(init_spaces), action_names(init_action_names),
//              spaces_names(init_spaces_names), actions_describe(init_actions_describe),
//              spaces_describe(init_spaces_describe) {}
    env(torch::Tensor _min,torch::Tensor _max,RLEngine::torchNet net,torch::Tensor gap):_min(_min),_max(_max),gap(gap) ,net(net){
}
    void train(torch::Tensor input);
    void get_runtime(torch::Tensor input);
private:
//    std::vector<std::vector<dataT>> actions;
//    std::vector<std::vector<dataT>> spaces;
//    std::vector<std::string> action_names;
//    std::vector<std::string> spaces_names;
//    std::vector<std::vector<std::string>> actions_describe;
//    std::vector<std::vector<std::string>> spaces_describe;
    torch::Tensor _min;
    torch::Tensor _max;
    torch::Tensor gap;

    torchNet net;
    RLEngine::Collection collection;

};
}
}