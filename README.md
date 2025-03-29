# PyTorch-Extensions-for-GPU-CPU
This repo contains PyTorch Extensions for 2D convolutional operations written wiht C++, OpenGL, OPpenMP, and Boost Compute.
The motivation behind writting faster extensions is that that currently PyTorch native 2D convolution is implmenetd using CuDNN,but does not support integrated GPU support currently. Thus I have written extensions for running convolution operations on CPU, integrated GPU and shared CPU/GPU.

**Hardware: Intel(R) Core(TM) i7-14700HX (Raptor Lake), Intel(R) UHD Graphics**

*Test size: 4096x4096 tensor Input, 256x356 tensor Filter*

  ### 1. iGPU [OpenCL]:
      Native PyTorch runtime: 0.30904 secs
      Extension runtime: 0.05667 secs
      Final Speedup: ~88%

  ### 2. CPU [OpenMP] :
      Native PyTorch runtime: 0.30904 secs
      Extension runtime: 0.6036 secs 
      Final Speedup: No increase

  ### 2. Shared iGPU-CPU [3:1] [OpenMP/OpenCL] :
      Native PyTorch runtime: 0.30904 secs
      Extension runtime: 0.28835 secs 
      Final Speedup: ~13%
      
