#define CL_TARGET_OPENCL_VERSION 300
#include <iostream>
#include <vector>
#include <torch/extension.h>
#include <omp.h>

float convolution(const float *input, const float *filter, int row, int col, 
                  int irows, int icols, int frows, int fcols, int rows_pad, int cols_pad) {
    float sum = 0.0f;
    if (row - rows_pad >= 0 && row + rows_pad < irows && col - cols_pad >= 0 && col + cols_pad < icols) {
        for (int k = 0; k < frows; k++) {
            for (int l = 0; l < fcols; l++) {
                int r = row + k - rows_pad;
                int c = col + l - cols_pad;
                sum += input[r * icols + c] * filter[k * fcols + l];
            }
        }
    }
    return sum;
}

torch::Tensor para_cpu(torch::Tensor &input, torch::Tensor &filter) {
    // Get dimensions
    int irows = input.size(0), icols = input.size(1);
    int frows = filter.size(0), fcols = filter.size(1);
    
    // Get padding
    int rows_pad = frows / 2;
    int cols_pad = fcols / 2;

    // Initialize output tensor
    torch::Tensor output = torch::zeros({irows, icols}, torch::kFloat);

    // Parallelized CPU workload using OpenMP
    #pragma omp parallel for collapse(2) //schedule(auto)
    for (int row = 0; row < irows; row++) {
        for (int col = 0; col < icols; col++) {
            output.data_ptr<float>()[row * icols + col] = 
                convolution(input.data_ptr<float>(), filter.data_ptr<float>(), row, col, 
                            irows, icols, frows, fcols, rows_pad, cols_pad);
        }
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("para_cpu", &para_cpu, "Optimized 2D convolution using OpenMP");
}
