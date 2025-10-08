# setup.py
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys, os, platform
import numpy as np

# ---- Platform helpers -------------------------------------------------------
IS_DARWIN = sys.platform == "darwin"
IS_LINUX  = sys.platform.startswith("linux")

# Homebrew prefixes
HB_PREFIXS = ["/opt/homebrew", "/usr/local"]  # M1/M2 vs Intel mac
HB_INCLUDE = [p + "/include" for p in HB_PREFIXS]
HB_LIB     = [p + "/lib" for p in HB_PREFIXS]

def omp_compile_args():
    if IS_DARWIN:
        # Apple Clang: use -Xpreprocessor -fopenmp and include path to libomp headers
        return ["-Xpreprocessor", "-fopenmp"]
    else:
        return ["-fopenmp"]

def omp_link_args():
    if IS_DARWIN:
        # Link against libomp on macOS
        return ["-lomp"]
    else:
        return ["-fopenmp"]

def brew_paths_if_exist():
    incs, libs = [], []
    for p in HB_INCLUDE:
        if os.path.isdir(p):
            incs.append(p)
    for p in HB_LIB:
        if os.path.isdir(p):
            libs.append(p)
    return incs, libs

brew_includes, brew_libs = brew_paths_if_exist()

# Intel compiler bits are generally irrelevant on macOS; keep Linux branch
intel = False
try:
    for line in open("newanalysis/voro/voro++-0.4.6/config.mk"):
        if "icc" in line:
            intel = True
except FileNotFoundError:
    pass

if intel and IS_LINUX:
    intel_lib_dir = ['/opt/intel/composer_xe_2015/lib/intel64/']
    intel_link_args = ['-lirc', '-limf']
else:
    intel_lib_dir = []
    intel_link_args = []

# Common include/library dirs
numpy_inc = np.get_include()

# If you installed FFTW and Voro++ yourself, keep your local paths first
fftw_inc = ['newanalysis/helpers/fftw-3.3.4/install/include']
fftw_lib = ['newanalysis/helpers/fftw-3.3.4/install/lib']

voro_inc = ['newanalysis/voro/voro++-0.4.6/install/include', 'newanalysis/voro']
voro_lib = ['newanalysis/voro/voro++-0.4.6/install/lib']

# On macOS, also search Homebrew prefixes for headers/libs
extra_includes = [numpy_inc] + (brew_includes if IS_DARWIN else [])
extra_lib_dirs = (brew_libs if IS_DARWIN else [])

# Some macOS setups prefer dynamic libs; if your static .a files aren’t present, fall back to -l flags
fftw_static = os.path.exists('newanalysis/helpers/fftw-3.3.4/install/lib/libfftw3.a') and not IS_DARWIN
voro_static = os.path.exists('newanalysis/voro/voro++-0.4.6/install/lib/libvoro++.a')

extensions = [
    Extension(
        'newanalysis.correl',
        sources=[
            'newanalysis/helpers/correl.pyx',
            'newanalysis/helpers/mod_Correl.cpp',
            'newanalysis/helpers/BertholdHorn.cpp'
        ],
        language='c++',
        include_dirs=fftw_inc + [numpy_inc],
        library_dirs=fftw_lib + extra_lib_dirs,
        extra_objects=(['newanalysis/helpers/fftw-3.3.4/install/lib/libfftw3.a'] if fftw_static else []),
        libraries=([] if fftw_static else ['fftw3']),
        extra_compile_args=omp_compile_args(),
        extra_link_args=omp_link_args(),
    ),
    Extension(
        'newanalysis.helpers',
        sources=['newanalysis/helpers/helpers.pyx', 'newanalysis/helpers/BertholdHorn.cpp'],
        language='c++',
        include_dirs=[numpy_inc] + extra_includes,
        extra_compile_args=omp_compile_args(),
        extra_link_args=omp_link_args(),
    ),
    Extension(
        'newanalysis.miscellaneous',
        sources=[
            'newanalysis/helpers/miscellaneous.pyx',
            'newanalysis/helpers/miscellaneous_implementation.cpp'
        ],
        language='c++',
        include_dirs=['newanalysis/helpers', numpy_inc] + extra_includes,
        extra_compile_args=omp_compile_args(),
        extra_link_args=omp_link_args(),
    ),
    Extension(
        'newanalysis.diffusion',
        sources=['newanalysis/helpers/diffusion.pyx'],
        language='c++',
        include_dirs=[numpy_inc] + extra_includes,
        extra_compile_args=omp_compile_args(),
        extra_link_args=omp_link_args(),
    ),
    Extension(
        'newanalysis.unfold',
        sources=['newanalysis/helpers/unfold.pyx', 'newanalysis/helpers/BertholdHorn.cpp'],
        language='c++',
        include_dirs=[numpy_inc] + extra_includes,
        extra_compile_args=omp_compile_args(),
        extra_link_args=omp_link_args(),
    ),
    Extension(
        'newanalysis.voro',
        sources=['newanalysis/voro/voro.pyx', 'newanalysis/voro/mod_voro.cpp'],
        language='c++',
        include_dirs=voro_inc + [numpy_inc] + extra_includes,
        library_dirs=voro_lib + intel_lib_dir + extra_lib_dirs,
        extra_objects=(['newanalysis/voro/voro++-0.4.6/install/lib/libvoro++.a'] if voro_static else []),
        libraries=([] if voro_static else ['voro++']),
        extra_compile_args=omp_compile_args(),
        extra_link_args=omp_link_args() + intel_link_args,
    ),
    Extension(
        'newanalysis.gfunction',
        sources=['newanalysis/gfunction/gfunction.pyx'],
        language='c++',
        include_dirs=[numpy_inc] + extra_includes,
        extra_compile_args=omp_compile_args(),
        extra_link_args=omp_link_args(),
    ),
    Extension(
        'newanalysis.functions',
        sources=['newanalysis/functions/py_functions.py'],
        language='c++'
    ),
]

setup(
    name='newanalysis',
    version='0.1dev',
    license='None',
    ext_modules=extensions,
    cmdclass={'build_ext': build_ext},
)

