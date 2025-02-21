#pragma once
#include "include.h"
#include "iostream"
using namespace std;
namespace tensorFly{
    namespace RLEngine{
        class Utils{
        public:
            static std::vector<std::vector<size_t>> get_spaces(
                    std::vector<std::pair<size_t, size_t>> ranges,
                    std::vector<std::size_t> gap
            ) {
                std::vector<std::vector<size_t>> all_choices(ranges.size()); // Initialize all_choices with correct size

                for (size_t i = 0; i < ranges.size(); ++i) {
                    size_t min = ranges[i].first;
                    size_t max = ranges[i].second;
                    size_t _gap = gap[i];

                    // Directly modify all_choices[i]
                    for (size_t j = min; j <= max; j += _gap) {
                        all_choices[i].push_back(j);
                    }
                }
                return all_choices;
            }
            static std::vector<std::vector<size_t>> getRandomValues(const std::vector<std::vector<size_t>>& vec, size_t numSamples) {
                std::vector<std::vector<size_t>> randomValuesList;
                for (size_t i = 0; i < numSamples; ++i) {
                    std::vector<size_t> randomValues;
                    for (const auto& innerVec : vec) {
                        if (!innerVec.empty()) {
                            size_t randomIndex = rand() % innerVec.size();
                            randomValues.push_back(innerVec[randomIndex]);
                        }
                    }
                    randomValuesList.push_back(randomValues);
                }
                return randomValuesList;
            }
            static  std::vector<std::vector<std::size_t>> inferActions (std::vector<std::size_t>& gap){
                std::vector<std::vector<size_t>> returnValuesList;
                for (auto _g = 0; _g < gap.size();_g++){
                    returnValuesList[_g].push_back(gap[_g]);
                    returnValuesList[_g].push_back(0);
                    returnValuesList[_g].push_back(-gap[_g]);
                }
                return returnValuesList;
            }
        };
    }
}