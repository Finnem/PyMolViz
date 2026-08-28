from setuptools import find_packages, setup
setup(name='pymolviz',
        version='1.3.2',
        description='Library to facilitate creation of PyMOL Vizualizations.',
        url='https://github.com/Finnem/PyMolViz',
        author='Finn Mier',
        license='MIT',
        packages=find_packages(),
        install_requires=[
                'numpy',
                'matplotlib',
                'scipy',
                'pandas',
                'cmap',
                'tqdm',
                'seaborn'
        ],
        long_description='See https://github.com/Finnem/PyMolViz for a detailed documentation.',
        zip_safe=False)

