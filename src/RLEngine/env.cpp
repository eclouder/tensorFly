#pragma once
#include "RLEngine/env.h"
#include "Utils.h"
#include "codegen/template.h"
using namespace tensorFly;
using namespace chrono;

void RLEngine::env::get_runtime(torch::Tensor input) {
    auto mul_c = tensorFly::codeInit::CPU_GEMM();
    for (int i = 0; i < input.size(0); ++i) {
        auto tensor = mul_c.get_torch_random_input();
        float *ptr1 = std::get<0>(tensor);
        float *ptr2 = std::get<1>(tensor);
        float *ptr3 = std::get<2>(tensor);

    }
}
void RLEngine::env::train(torch::Tensor input) {
    auto actions = net.action_module.forward(input);
    auto values = net.value_module.forward(input);
    collection.states.push_back(input);
    collection.actions.push_back(actions);
    torch::Tensor input_transform = input * (_max - _min) + _min;
    std::cout << actions.sizes() << std::endl;
    auto _action = sample_from_categorical(actions).squeeze();
    torch::Tensor next_state = torch::zeros_like(input);
    torch::Tensor runtime = torch::zeros({input.size(0)});
    for (int i = 0; i < _action.size(0); ++i) {
        auto action_id = _action[i].item<int>();
        auto _cur_step = input[0];
        auto _input = input_transform[i];
        switch (action_id) {
            case 0:
                if (_input[0].item<float>() >= (_min[0].item<float>() + gap[0].item<float>())) {
                    next_state[i][0] = input_transform[i][0] - gap[0];
                }
                if (_input[1].item<float>() >= (_min[1].item<float>() + gap[1].item<float>())) {
                    next_state[i][1] = input_transform[i][1] - gap[1];
                }
                break;

            case 1:
                if (_input[0].item<float>() >= (_min[0].item<float>() + gap[0].item<float>())) {
                    next_state[i][0] = input_transform[i][0] - gap[0];
                }
                break;

            case 2:
                if (_input[0].item<float>() >= (_min[0].item<float>() + gap[0].item<float>())) {
                    next_state[i][0] = input_transform[i][0] - gap[0];
                }
                if (_input[1].item<float>() <= (_max[1].item<float>() - gap[1].item<float>())) {
                    next_state[i][1] = input_transform[i][1] +gap[1];
                }
                break;

            case 3:
                if (_input[1].item<float>() >= (_min[1].item<float>() + gap[1].item<float>())) {
                    next_state[i][1] = input_transform[i][1] - gap[1];
                }
                break;

            case 4:

                break;

            case 5:
                if (_input[1].item<float>() <= (_max[1].item<float>() - gap[1].item<float>())) {
                    next_state[i][1] = input_transform[i][1] +gap[1];
                }
                break;
            case 6:
                if (_input[0].item<float>() <= (_max[0].item<float>() - gap[0].item<float>())) {
                    next_state[i][0] = input_transform[i][0] +gap[0];
                }
                if (_input[1].item<float>() >= (_min[1].item<float>() + gap[1].item<float>())) {
                    next_state[i][1] = input_transform[i][1] - gap[1];
                }
            case 7:
                if (_input[0].item<float>() <= (_max[0].item<float>() - gap[0].item<float>())) {
                    next_state[i][0] = input_transform[i][0] +gap[0];
                }
            case 8:
                if (_input[0].item<float>() <= (_max[0].item<float>() - gap[0].item<float>())) {
                    next_state[i][0] = input_transform[i][0] +gap[0];
                }
                if (_input[1].item<float>() <= (_max[1].item<float>() - gap[1].item<float>())) {
                    next_state[i][1] = input_transform[i][1] +gap[1];
                }
        }
        auto mul_c = tensorFly::codeInit::CPU_GEMM();
        auto tensor = mul_c.get_torch_random_input();
        float* ptr1 = std::get<0>(tensor);
        float* ptr2 = std::get<1>(tensor);
        float* ptr3 = std::get<2>(tensor);
        auto start = system_clock::now();
        mul_c.matmulImplRowColParallelInnerTiling<1024,1024,1024,next_state[i][0].item<int>(),2,next_state[i][0].item<int>()>(ptr1,ptr2,ptr3);
        auto finish = system_clock::now();
        auto duration = duration_cast<microseconds>(finish - start);
        auto cost = double(duration.count()) * microseconds::period::num / microseconds::period::den;
        runtime[i] = cost;
    }
}
