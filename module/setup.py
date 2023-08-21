from setuptools import setup
setup(name='pymolviz',
        version='1.1.3',
        description='Library to facilitate creation of PyMOL Vizualizations.',
        url='https://github.com/Finnem/PyMolViz',
        author='Finn Mier',
        license='MIT',
        packages=['pymolviz.meshes', 'pymolviz.PyMOLobjects', 'pymolviz.util', 'pymolviz.volumetric', 'pymolviz'],
        install_requires=[
                'numpy',
                'matplotlib',
                'scipy',
                'pandas'
        ],
        long_description='See https://github.com/Finnem/PyMolViz for a detailed documentation.',
        zip_safe=False)
