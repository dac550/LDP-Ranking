from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ldp-poison-toolkit",
    version="1.0.0",
    author="LDP Poison Toolkit Contributors",
    description="Poisoning attack evaluation framework for Local Differential Privacy ranking estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21",
        "pandas>=1.3",
        "scipy>=1.7",
        "xxhash>=3.0",
        "scikit-learn>=1.0",
        "matplotlib>=3.4",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "pytest-cov", "black", "isort", "mypy"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords=[
        "local differential privacy",
        "poisoning attacks",
        "ranking estimation",
        "security",
        "privacy",
        "federated learning",
    ],
)
