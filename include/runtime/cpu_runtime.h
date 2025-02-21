#pragma once
# include "include.h"
#include "chrono"

namespace tensorFly{
    namespace runtime{
        class CpuRuntime{
            CpuRuntime(std::function<void (float*, float*,float *)> func):func(func){
            }
        private:
            std::function<void (float*, float*,float *)> func;

        public:
        };
    }
}