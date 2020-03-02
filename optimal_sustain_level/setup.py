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
    Extension("sustained_ihp_c", ["sustained_ihp_c.pyx"], include_dirs=[numpy.get_include()])
]
setup(
    name='sustained_ihp_c',
    ext_modules=cythonize(extensions),
    script_args=["build_ext", "--inplace"]
)