from distutils.core import setup, Extension
from distutils.ccompiler import new_compiler
from Cython.Distutils import build_ext
import numpy

intel = False
for line in open("src/voro/voro++-0.4.6/config.mk"):
    if "icc" in line:
        intel = True

if intel:
    intel_lib_dir = ['/opt/intel/composer_xe_2015/lib/intel64/']
    intel_link_args = ['-lirc','-limf']
else:
    intel_lib_dir = []
    intel_link_args = []



setup(
  name = 'newanalysis',
    version='0.1dev',
    license='None',
    long_description=open('../README').read(),
  ext_modules=[
      Extension('newanalysis.correl',
                sources=['src/helpers/correl.pyx','src/helpers/mod_Correl.cpp', 'src/helpers/BertholdHorn.cpp'],
                language='c++',
                include_dirs = ['src/helpers/fftw-3.3.4/install/include', numpy.get_include()],
                library_dirs = ['src/helpers/fftw-3.3.4/install/lib'],
                extra_objects = ['src/helpers/fftw-3.3.4/install/lib/libfftw3.a'],
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp']),
      Extension('newanalysis.helpers',
                sources=['src/helpers/helpers.pyx','src/helpers/BertholdHorn.cpp'],
                language='c++',
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp'],
                include_dirs=[numpy.get_include()]),
      Extension('newanalysis.diffusion',
                sources=['src/helpers/diffusion.pyx'],
                language='c++',
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp'],
                include_dirs=[numpy.get_include()]),
       Extension('newanalysis.unfold',
                sources=['src/helpers/unfold.pyx','src/helpers/BertholdHorn.cpp'],
                language='c++',
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp'],
                include_dirs=[numpy.get_include()]),
       Extension('newanalysis.voro', 
                sources=['src/voro/voro.pyx','src/voro/mod_voro.cpp'],
                language='c++',
                include_dirs = ['src/voro/voro++-0.4.6/install/include','src/voro', numpy.get_include()],
                library_dirs = ['src/voro/voro++-0.4.6/install/lib']+intel_lib_dir,
                extra_objects = ['src/voro/voro++-0.4.6/install/lib/libvoro++.a'],
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp']+intel_link_args),
       Extension('newanalysis.gfunction',
                sources=['src/gfunction/gfunction.pyx'],
                language='c++',
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp'],
                include_dirs=[numpy.get_include()]),
      Extension('newanalysis.functions',
                sources=['src/functions/py_functions.py'],
                include_dirs=[numpy.get_include()]),
    ],
  cmdclass = {'build_ext': build_ext}
)
