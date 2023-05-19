from distutils.core import setup, Extension
from distutils.ccompiler import new_compiler
from Cython.Distutils import build_ext
import numpy

intel = False
for line in open("newanalysis/voro/voro++-0.4.6/config.mk"):
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
#    long_description=open('../README').read(),
  ext_modules=[
      Extension('newanalysis.correl',
                sources=['newanalysis/helpers/correl.pyx','newanalysis/helpers/mod_Correl.cpp', 'newanalysis/helpers/BertholdHorn.cpp'],
                language='c++',
                include_dirs = ['newanalysis/helpers/fftw-3.3.4/install/include', numpy.get_include()],
                library_dirs = ['newanalysis/helpers/fftw-3.3.4/install/lib'],
                extra_objects = ['newanalysis/helpers/fftw-3.3.4/install/lib/libfftw3.a'],
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp']),
      Extension('newanalysis.helpers',
                sources=['newanalysis/helpers/helpers.pyx', 'newanalysis/helpers/BertholdHorn.cpp'],
                language='c++',
                include_dirs=[numpy.get_include()],
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp']),
      Extension('newanalysis.miscellaneous',
               # ['newanalysis/helpers/miscellaneous.%s' % ("pyx" if use_cython else "cpp"),
                sources = [
                #'newanalysis/helpers/miscellaneous_newsyntax.pyx',
                #'newanalysis/helpers/miscellaneous.cpp',
                'newanalysis/helpers/miscellaneous.pyx',
                'newanalysis/helpers/miscellaneous_implementation.cpp'
                ],
                language='c++',
                include_dirs = [numpy.get_include(), 'newanalysis/helpers'],
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp'],
            ),
      Extension('newanalysis.diffusion',
                sources=['newanalysis/helpers/diffusion.pyx'],
                language='c++',
                include_dirs=[numpy.get_include()],
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp']),
       Extension('newanalysis.unfold',
                 sources=['newanalysis/helpers/unfold.pyx','newanalysis/helpers/BertholdHorn.cpp'],
                 language='c++',
                 include_dirs=[numpy.get_include()],
                 extra_compile_args=['-fopenmp'],
                 extra_link_args=['-fopenmp']),
       Extension('newanalysis.voro', 
                 sources=['newanalysis/voro/voro.pyx','newanalysis/voro/mod_voro.cpp'],
                 language='c++',
                 include_dirs = ['newanalysis/voro/voro++-0.4.6/install/include','newanalysis/voro', numpy.get_include()],
                 library_dirs = ['newanalysis/voro/voro++-0.4.6/install/lib']+intel_lib_dir,
                 extra_objects = ['newanalysis/voro/voro++-0.4.6/install/lib/libvoro++.a'],
                 extra_compile_args=['-fopenmp'],
                 extra_link_args=['-fopenmp']+intel_link_args),
       Extension('newanalysis.gfunction',
                 sources=['newanalysis/gfunction/gfunction.pyx'],
                 language='c++',
                 extra_compile_args=['-fopenmp'],
                 include_dirs=[numpy.get_include()],
                 extra_link_args=['-fopenmp']),
      Extension('newanalysis.functions',
                sources=['newanalysis/functions/py_functions.py']),
    ],
  cmdclass = {'build_ext': build_ext}
)
