from setuptools import setup
setup(name='pymolviz',
        version='0.1',
        description='Library to create PyMOL scripts which create CGO to display arbitrary graphics in PyMOL.',
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
