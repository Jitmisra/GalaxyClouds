from setuptools import setup, find_packages

setup(
    name='galaxyclouds',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.24',
        'pandas>=1.5',
        'matplotlib>=3.7',
        'scikit-learn>=1.3',
        'xgboost>=1.7',
        'shap>=0.42',
        'scipy>=1.10',
        'plotly>=5.14',
        'astropy>=5.3',
    ],
    description='Point cloud observable library for galaxy morphology analysis',
    author='Jit Misra',
)
