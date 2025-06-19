from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "mcts_cpp",
        ["mcts.cpp", "kalah_game.cpp", "neural_network.cpp", "python_bindings.cpp"],
        include_dirs=["/usr/local/include/torch", "/usr/local/cuda/include"],
        libraries=["torch", "torch_cuda", "c10", "c10_cuda", "pthread"],
        library_dirs=["/usr/local/lib", "/usr/local/cuda/lib64"],
        cxx_std=17,
        extra_compile_args=["-O3", "-march=native", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
]

setup(
    name="mcts_cpp",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
