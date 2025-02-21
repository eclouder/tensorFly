#pragma once

#include "string"
namespace tensorFly{
    namespace runtime{
        class nsight_compute_runtime{
        public:
            std::string run(std::string const path);
        };
    }
}