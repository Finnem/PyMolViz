from setuptools import setup
setup(name='pymolviz',
        version='1.0',
        description='Library to facilitate creation of PyMOL Vizualizations.',
        url='',
        author='Finn Mier',
        license='MIT',
        packages=['pymolviz'],
        install_requires=[
                'numpy',
                'matplotlib',
                'scipy',
        ],
        zip_safe=False)
