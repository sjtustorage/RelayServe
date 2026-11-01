from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='amx_gemm_bf16_ext',
    ext_modules=[
        CppExtension(
            'amx_gemm_bf16_ext',
            [
                'gemm_pack.cpp',
                'gemm_amx.cpp',
                'gemm_avx512.cpp',
                'gemm_dispatch.cpp',
                'gemm_bindings.cpp',
                'gemm_prefetch_amx.cpp'
            ],
            extra_compile_args={
                'cxx': ['-fopenmp', '-mavx512bf16', '-mamx-tile', '-mamx-bf16', '-mamx-int8', '-O3', '-std=c++20', '-mwaitpkg'],
            },
            extra_link_args=['-fopenmp', '-lnuma']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)