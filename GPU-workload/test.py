import torch
import gpu_conv2d_extension
import torch.utils.benchmark as benchmark

# Create large random tensors for input and filter
input = torch.rand(4096, 4096)
filter = torch.rand(8,8)

def work(a, b):
    return gpu_conv2d_extension.my_xpu_conv2d(a, b)

output = work(input, filter)

print(output)

time = benchmark.Timer(
    stmt='work(input, filter)',
    setup='from __main__ import work, input, filter',
    globals=globals()
).timeit(10)

print(f"Average execution time over 100       runs: {time.mean} seconds")
# print(output)