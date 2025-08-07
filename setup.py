from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="dynamic-recruitment-networks",
    version="0.1.0",
    author="Prashant Raju",
    author_email="rajuprashant@gmail.com",
    description="Biologically-inspired neural networks with dynamic neuron recruitment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/prashantcraju/Dynamic-Recruitment-Network-working",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=6.0", "black>=21.0", "flake8>=3.9", "mypy>=0.910"],
        "notebooks": ["jupyterlab>=3.0", "ipywidgets>=7.6"],
    },
)

