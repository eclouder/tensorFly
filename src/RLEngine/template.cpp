#include "RLEngine/utils.h"
#include "codegen/template.h"
using namespace tensorFly;
std::vector<std::vector<size_t>> tensorFly::codeInit::CPU_GEMM::get_spaces() {
    return tensorFly::RLEngine::Utils::get_spaces(this->spaces_range, this->spaces_gap);

}

std::vector<std::vector<size_t>> tensorFly::codeInit::CPU_GEMM::get_action_choices() {
    return tensorFly::RLEngine::Utils::inferActions(this->spaces_gap);
}