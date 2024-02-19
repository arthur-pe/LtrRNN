from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='ltrRNN',
    packages=find_packages(exclude=['tests*']),
    version='0.0.1',

    description='Package to perform low-tensor rank recurrent neural network decomposition of neural data over learning.',
    long_description=long_description,
    long_description_content_type='text/markdown',

    url='https://github.com/arthur-pe/ltrRNN',
    author='Arthur Pellegrino',
    license='MIT',
    install_requires=['torch',
                      'numpy',
                      'matplotlib',
                      'tqdm',
                      'scipy',
                      'pandas',
                      'PyYAML',
                      'torchsde',
                      'scikit-learn',
                      ],
    python_requires='>=3.10',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
    ],
)