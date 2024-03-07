import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mkor",
    version="0.1.2",
    author="Mohammad Mozaffari, Sikan Li, Zhao Zhang, Maryam Mehri Dehnavi",
    author_email="mmozaffari@cs.toronto.edu",
    description="PyTorch implementation of MKOR: Momentum-Enabled Kronecker-Factor-Based Optimizer Using Ranks-1 Updates",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mohammad-Mozaffari/mkor",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)