from setuptools import setup, find_packages

version = '0.1.3'


# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='nb_ocl',
    version=version,
    description='Newton Basins implementation in Python/OpenCL',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/gmagno/nb-ocl',
    author='gmagno',
    author_email='goncalo.magno@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3',
    ],
    setup_requires=['numpy', 'mako', 'pybind11', 'jinja2', 'matplotlib', 'pyopencl'],
    install_requires=['numpy', 'mako', 'pybind11', 'jinja2', 'matplotlib', 'pyopencl'],
    keywords='newton basins',
    project_urls={
        'Bug Reports': 'https://github.com/gmagno/nb-ocl/issues',
        'Source': 'https://github.com/gmagno/nb-ocl',
    },
)
