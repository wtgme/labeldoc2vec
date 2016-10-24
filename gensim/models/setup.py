from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension("labeldoc2vec_inner", ["labeldoc2vec_inner.pyx"])
]

setup(
    ext_modules = cythonize(extensions)
)