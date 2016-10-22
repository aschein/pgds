#!/usr/bin/env pythonv
import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize


setup(
    cmdclass={'build_ext': build_ext},
    include_dirs=[np.get_include(), '../../fatwalrus'],
    ext_modules=cythonize('**/*.pyx', include_path=['../../fatwalrus'])
)
