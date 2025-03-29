from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='para_cpu',
    ext_modules=[
        CppExtension(name='para_cpu',
                                 sources=['para_cpu.cpp'])
    ],
    
    cmdclass={
        'build_ext': BuildExtension
    }
)
