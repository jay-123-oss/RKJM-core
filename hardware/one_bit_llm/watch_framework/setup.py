import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension

here = os.path.abspath(os.path.dirname(__file__))

sources = [
    os.path.join(here, "watch_framework", "csrc", "watch_grid.cpp")
]

include_dirs = [
    os.path.join(here, "watch_framework", "csrc")
]

extra_compile_args = {
    "cxx": ["-O3", "-std=c++20", "-fopenmp", "-Wall", "-Wextra"]
}

extra_link_args = ["-fopenmp"]

setup(
    name="watch_framework",
    version="0.1.0",
    description="PyTorch Bitwise CSA 2D Watch Grid Engine and 1-Bit QAT Framework",
    author="2D Watch Grid Team",
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            name="watch_framework._C",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(use_ninja=True)
    },
    install_requires=[
        "torch",
        "numpy",
    ],
    python_requires=">=3.8",
)
