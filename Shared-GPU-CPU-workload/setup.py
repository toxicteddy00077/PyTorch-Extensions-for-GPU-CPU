from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='gpu_conv2d_extension',
    ext_modules=[CppExtension(
        'gpu_conv2d_extension', ['gpu_conv2d_extension.cpp'],
        extra_compile_args=['-std=c++17', '-fsycl'],
        extra_link_args=['-ldnnl']
    )],
    
    cmdclass={
        'build_ext': BuildExtension
    }
)
