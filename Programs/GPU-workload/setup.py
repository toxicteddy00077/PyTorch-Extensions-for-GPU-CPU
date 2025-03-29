from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='gpu_conv2d_extension',
    ext_modules=[
        CppExtension(name='gpu_conv2d_extension',
                                 sources=['gpu_conv2d_extension.cpp'],
                                 extra_link_args=["-lOpenCL"])
    ],
    
    cmdclass={
        'build_ext': BuildExtension
    }
)
