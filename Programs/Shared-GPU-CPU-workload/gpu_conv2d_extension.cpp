#include <torch/extension.h>
#include "dnnl.hpp"
#include <unordered_map>

using namespace dnnl;

torch::Tensor my_xpu_conv2d(torch::Tensor input, torch::Tensor filter) {
    // Ensure input tensors are contiguous and of type float32.
    input = input.contiguous().to(torch::kFloat);
    filter = filter.contiguous().to(torch::kFloat);

    // Input shape: [H, W], filter shape: [fH, fW]
    int H = input.size(0);
    int W = input.size(1);
    int fH = filter.size(0);
    int fW = filter.size(1);

    // Calculate padding for 'same' convolution
    int pad_h_total = fH - 1;
    int pad_h_left = pad_h_total / 2;
    int pad_h_right = pad_h_total - pad_h_left;
    int pad_w_total = fW - 1;
    int pad_w_left = pad_w_total / 2;
    int pad_w_right = pad_w_total - pad_w_left;

    // Output dimensions remain the same as input
    int outH = H;
    int outW = W;

    // Create engine and stream
    engine eng;
    bool use_gpu = engine::get_count(engine::kind::gpu) > 0;
    if (use_gpu)
        eng = engine(engine::kind::gpu, 0);
    else
        eng = engine(engine::kind::cpu, 0);
    stream s(eng);

    // Move tensors to the correct device
    torch::Device device = use_gpu ? torch::kXPU : torch::kCPU;
    input = input.to(device);
    filter = filter.to(device);

    // Add batch and channel dimensions
    auto input_4d = input.unsqueeze(0).unsqueeze(0); // [1, 1, H, W]
    auto filter_4d = filter.unsqueeze(0).unsqueeze(0); // [1, 1, fH, fW]

    // Memory dimensions and parameters
    memory::dims src_dims = {1, 1, H, W};
    memory::dims weights_dims = {1, 1, fH, fW};
    memory::dims dst_dims = {1, 1, outH, outW};
    memory::dims strides = {1, 1};
    memory::dims dilations = {1, 1};
    memory::dims padding_l = {pad_h_left, pad_w_left};
    memory::dims padding_r = {pad_h_right, pad_w_right};

    // User memory descriptors (NCHW and OIHW formats)
    auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
    auto weights_md = memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::oihw);
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nchw);

    // Create convolution descriptor
    auto conv_desc = convolution_forward::primitive_desc(eng,
        prop_kind::forward_inference,
        algorithm::convolution_direct,
        src_md,
        weights_md,
        dst_md,
        strides,
        dilations,
        padding_l,
        padding_r);

    auto conv_prim = dnnl::convolution_forward(conv_desc);

    auto src_user = memory(src_md, eng, input_4d.data_ptr());
    auto weights_user = memory(weights_md, eng, filter_4d.data_ptr());

    // Create and reorder memory objects
    auto src_prim = memory(conv_desc.src_desc(), eng);
    reorder(src_user, src_prim).execute(s, src_user, src_prim);

    auto weights_prim = memory(conv_desc.weights_desc(), eng);
    reorder(weights_user, weights_prim).execute(s, weights_user, weights_prim);

    auto dst_prim = memory(conv_desc.dst_desc(), eng);

    std::unordered_map<int, dnnl::memory> conv_args;
    conv_args.insert({DNNL_ARG_SRC, src_prim});
    conv_args.insert({DNNL_ARG_WEIGHTS, weights_prim});
    conv_args.insert({DNNL_ARG_DST, dst_prim});

    conv_prim.execute(s, conv_args);
    s.wait();

    // Prepare output tensor and reorder back to user format
    torch::Tensor output = torch::empty({1, 1, outH, outW}, input.options());
    auto dst_user = memory(dst_md, eng, output.data_ptr());
    reorder(dst_prim, dst_user).execute(s, dst_prim, dst_user);
    s.wait();

    return output.squeeze();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_xpu_conv2d", &my_xpu_conv2d, "2D convolution using oneDNN (GPU/CPU)");
}