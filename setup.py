from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="automl-pipeline",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="An automated machine learning pipeline for classification and regression tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/automl-pipeline",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.1",
            "black>=21.12b0",
            "flake8>=4.0.1",
            "isort>=5.10.1",
            "mypy>=0.930",
        ],
        "docs": [
            "sphinx>=4.2.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.12.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "machine learning",
        "automl",
        "automated machine learning",
        "data science",
        "data preprocessing",
        "model selection",
        "hyperparameter optimization",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/automl-pipeline/issues",
        "Source": "https://github.com/yourusername/automl-pipeline",
    },
    include_package_data=True,
    zip_safe=False,
)
