from setuptools import setup
import versioneer

requirements = [
    'tensorflow>=2.0',
    'numpy',
    'pandas', 'scipy',                      # Data
    'colorlog',                             # Logging
    'matplotlib', 'seaborn', 'plotly'       # Plotting
]

setup(
    name='cvnn',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Library to help implement a complex-valued neural network (cvnn) using tensorflow as back-end",
    license="MIT",
    author="J Agustin BARRACHINA",
    author_email='joseagustin.barra@gmail.com',
    url='https://github.com/NEGU93/cvnn',
    packages=['cvnn'],
    entry_points={
        'console_scripts': [
            'cvnn=cvnn.cli:cli'
        ]
    },
    install_requires=requirements,
    keywords='cvnn',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)
