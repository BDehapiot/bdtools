from setuptools import setup, find_packages

setup(
    name='bd-tools',
    version='0.2.3',
    packages=find_packages(),
    description='Collection of tools for recurring tasks',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Benoit Dehapiot',
    author_email='b.dehapiot@gmail.com',
    license='GNU General Public License v3 (GPLv3)',
    install_requires=[
        "numpy~=1.24.0",
        "scipy",
        "scikit-image",
        "joblib",
        "numba",
    ],
    extras_require={
        "tf-gpu": [
            "napari[all]",
            "albumentations",
            "tensorflow-gpu==2.10",
        ],
        "tf-nogpu": [
            "napari[all]",
            "albumentations",
            "tensorflow==2.10",
        ],
    },
    python_requires='>=3.9',
)
