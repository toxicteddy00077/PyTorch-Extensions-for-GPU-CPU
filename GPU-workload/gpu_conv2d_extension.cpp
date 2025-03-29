#define CL_TARGET_OPENCL_VERSION 300
#include <torch/extension.h>
#include <boost/compute.hpp>
#include <vector>

torch::Tensor my_xpu_conv2d(torch::Tensor &input, torch::Tensor &filter) {
    namespace compute = boost::compute;

    //get device, create context and command queue
    compute::device device = compute::system::devices()[1];
    compute::context context(device); 
    compute::command_queue queue(context, device);

    //get dimension
    int irows = input.size(0);int icols = input.size(1);
    int frows = filter.size(0);int fcols = filter.size(1);

    //get padding
    int rows_pad = frows/2;
    int cols_pad = fcols/2;

    //initialize the output tensor
    torch::Tensor output = torch::zeros({irows, icols}, torch::kFloat);

    //allocate on device
    compute::vector<float> d_input(irows*icols, context);
    compute::vector<float> d_filter(frows*fcols, context);
    compute::vector<float> d_output(irows*icols, context);

    //copy data to device
    compute::copy(input.data_ptr<float>(), input.data_ptr<float>() + irows*icols, d_input.begin(), queue);
    compute::copy(filter.data_ptr<float>(), filter.data_ptr<float>() + frows*fcols, d_filter.begin(), queue);

    //opencl kernel source passed as raw bytes
    const char src[] = R"(
    __kernel void my_xpu_conv2d(
        __global const float* input,
        __global const float* filter,
        __global float* output,
        int irows, int icols,
        int frows, int fcols,
        int rows_pad, int cols_pad) {

        int row = get_global_id(0);
        int col = get_global_id(1);

        if (row < irows && col < icols) {
            float sum = 0.0;

            if(row - rows_pad >= 0 && row + rows_pad < irows && col - cols_pad >= 0 && col + cols_pad < icols){
                for (int k = 0; k < frows; k++) {
                    for (int l = 0; l < fcols; l++) {

                        int r = row + k - rows_pad;
                        int c = col + l - cols_pad;

                        sum += input[r * icols + c] * filter[k * fcols + l];

                    }
                }
            }
            output[row * icols + col] = sum;
        }
    })";

    //compile 
    compute::program program = compute::program::build_with_source(src, context, "-cl-std=CL3.0");
    compute::kernel kernel = program.create_kernel("my_xpu_conv2d");

    //set kernel arguments
    kernel.set_arg(0, d_input.get_buffer());
    kernel.set_arg(1, d_filter.get_buffer());
    kernel.set_arg(2, d_output.get_buffer());
    kernel.set_arg(3, irows);kernel.set_arg(4, icols);
    kernel.set_arg(5, frows);kernel.set_arg(6, fcols);
    kernel.set_arg(7, rows_pad);
    kernel.set_arg(8, cols_pad);

    //execute kernel
    size_t local_work[2] = {16, 16};
    size_t global_work[2] = {static_cast<size_t>(irows), static_cast<size_t>(icols)};//static cast is neccessary since the enqueue_nd_range_kernel function expects size_t
    queue.enqueue_nd_range_kernel(kernel, 2, NULL, global_work,local_work);
    queue.finish();

    //copy result back to host
    compute::copy(d_output.begin(), d_output.end(), output.data_ptr<float>(), queue);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_xpu_conv2d", &my_xpu_conv2d, "convolution 2d using OpenCL and Boost Compute");
}
