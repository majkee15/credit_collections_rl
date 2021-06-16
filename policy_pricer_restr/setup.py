from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import os


def find_pyx(path='.'):
    pyx_files = []
    for root, dirs, filenames in os.walk(path):
        for fname in filenames:
            if fname.endswith('.pyx'):
                pyx_files.append(os.path.join(root, fname))
    return pyx_files


extensions = [
    Extension("cython_pricer.cython_pricer_naive", [r"cython_pricer/cython_pricer_naive.pyx"], include_dirs=[numpy.get_include()],
              extra_compile_args=['/openmp'],
              extra_link_args=['/openmp']),
    Extension("cython_pricer.cython_pricer_optimized", [r"cython_pricer/cython_pricer_optimized.pyx"],
              include_dirs=[numpy.get_include()], extra_compile_args=['/openmp'],
              extra_link_args=['/openmp'])
]
# os.environ["CC"] = "g++-9"
# os.environ["CXX"] = "g++-9"
setup(
    name='cython_pricer',
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),#, build_dir=r"cython_pricer/build"),
    script_args=["build_ext", "--inplace"],
)
