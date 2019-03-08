from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bmm',
    ext_modules=[
        CUDAExtension('bmm', [
            'bmm.cpp',
            'bmm_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
