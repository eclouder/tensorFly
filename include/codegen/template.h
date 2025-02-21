#include <ATen/ops/rand.h>
#include "include.h"
#include "torch/torch.h"
#include "RLEngine/utils.h"
using namespace tensorFly;
namespace tensorFly{
    namespace codeInit{
        class CPU_GEMM{
        public:
            template <int rows, int columns, int inners,
                    int tileSize ,int collapse_size,int rol_or_col_tile_size>
            inline void matmulImplRowColParallelInnerTiling(const float *left,
                                                            const float *right,
                                                            float *result) {
            #pragma omp parallel for shared(result, left, right) default(none) \
              collapse(2) num_threads(24)
                            for (int rowTile = 0; rowTile < rows; rowTile += rol_or_col_tile_size) {
                                for (int columnTile = 0; columnTile < columns; columnTile += rol_or_col_tile_size) {
                                    for (int innerTile = 0; innerTile < inners; innerTile += tileSize) {
                                        for (int row = rowTile; row < rowTile + rol_or_col_tile_size; row++) {
                                            int innerTileEnd = std::min(inners, innerTile + tileSize);
                                            for (int inner = innerTile; inner < innerTileEnd; inner++) {
                                                for (int col = columnTile; col < columnTile + rol_or_col_tile_size; col++) {
                                                    result[row * columns + col] +=
                                                            left[row * inners + inner] * right[inner * columns + col];
                                                } } } } } } }
            std::tuple<float*, float*, float*> get_torch_random_input(size_t row = 1024,size_t col = 1024,size_t k = 1024){
                auto left_tensor = torch::rand({static_cast<int>(row), static_cast<int>(k)});
                auto right_tensor = torch::rand({static_cast<int>(k), static_cast<int>(col)});
                auto result_tensor = torch::rand({static_cast<int>(row), static_cast<int>(col)});
                float* left = left_tensor.data_ptr<float>();
                float* right = right_tensor.data_ptr<float>();
                float* result = result_tensor.data_ptr<float>();

                return std::make_tuple(left, right, result);
            }
            std::vector<std::vector<size_t>> get_spaces();
            std::vector<std::vector<size_t>> get_action_choices();
            auto get_spaces_range(){return spaces_range;};
            size_t get_action_size(){return action_size;};
            auto get_space_gap(){return spaces_gap;};
            size_t get_spaces_size(){return spaces_range.size();};
        private:
            size_t action_size = 6;
            std::vector<std::pair<size_t,size_t>> spaces_range = {
                    {16,256},
                    {16,256}
            };
            std::vector<std::size_t> spaces_gap = {
                    4,4
            };

        };

        }

        ;

    }
