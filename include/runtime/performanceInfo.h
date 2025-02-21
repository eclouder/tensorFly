#pragma once
#include "map"
#include "string"
#include "variant"

namespace tensorFly{
namespace runtime{
    class performanceInfo{
    public:
        virtual void parseReport(const std::string& reportContent);
    private:
        std::map<std::string,std::variant<size_t,float_t>> data_map;
    };
    class ncuInfo:performanceInfo{

    };
}
}