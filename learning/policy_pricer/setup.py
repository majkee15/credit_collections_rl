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
    Extension("cython_pricer_naive", ["cython_pricer_naive.pyx"], include_dirs=[numpy.get_include()]),
    Extension("cython_pricer_optimized", ["cython_pricer_optimized.pyx"], include_dirs=[numpy.get_include()])
]
# os.environ["CC"] = "g++-9"
# os.environ["CXX"] = "g++-9"
setup(
    name='cython_pricer',
    ext_modules=cythonize(extensions, compiler_directives={'language_level' : "3"}),
    script_args=["build_ext", "--inplace"],
)
