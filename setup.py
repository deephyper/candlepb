from setuptools import setup


install_requires=[
        'pandas',
        'console-menu',
        'numpy',
        'matplotlib'
]

setup(
  name = 'candlepb',
  packages = ['candlepb'],
  install_requires=install_requires,
  entry_points = {
        'console_scripts': [
            'candle-plot=candlepb.exp.graph:main',
            ],
    }
)