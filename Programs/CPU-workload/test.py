import torch
import para_cpu
import torch.utils.benchmark as benchmark

input = torch.rand(1024, 1024)
filter = torch.rand(256, 256)

def work(input, filter):
    return para_cpu.para_cpu(input, filter)

output = work(input, filter)

print(output)


time = benchmark.Timer(
    stmt='work(input, filter)',
    setup='from __main__ import work, input, filter',
    globals=globals()
).timeit(1)

print(f"Average execution time over 10 runs: {time.mean} seconds")
# print(output)