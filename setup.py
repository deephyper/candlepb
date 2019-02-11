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
)