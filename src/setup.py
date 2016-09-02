import numpy as np
from Cython.Build import cythonize
from distutils.core import setup


setup(
    include_dirs=[np.get_include()],
    ext_modules=cythonize(['**/*.pyx'])
)
