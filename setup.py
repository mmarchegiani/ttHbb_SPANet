from setuptools import find_packages, setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="ttHbb_SPANet",
    packages=find_packages(),
    #version="2.2.0",
    description="Repository for development of a signal vs background classifier in the ttH(bb) analysis based on SPANet.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Matteo Marchegiani",
    author_email="matteo.marchegiani@cern.ch",
    url="https://github.com/mmarchegiani/ttHbb_SPANet",
    #license="BSD-3-Clause",
    install_requires=[
        'torch>=2.0',
        'numpy>=1.24',
        'h5py>=3.9',
        'numba>=0.57',
        'pytorch-lightning>=2.0',
        'tqdm',
        'sympy',
        'pyyaml',
        'opt_einsum',
        'scikit-learn',
        'rich',
        'htcondor',
        'law'
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Environment :: GPU :: NVIDIA CUDA :: 11.7',
        'Environment :: GPU :: NVIDIA CUDA :: 11.8',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Physics',
        'Typing :: Typed'
    ],
)
