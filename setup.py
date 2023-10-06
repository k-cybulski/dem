from setuptools import find_packages, setup

with open("README.md", "r") as file_:
    long_description = file_.read()

setup(
    name="dem",
    version="0.1.0",
    author="Krzysztof Cybulski",
    author_email="",
    description="Dynamic Expectation Maximization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/k-cybulski/declair",
    packages=find_packages(exclude=("tests", "examples")),
    python_requires=">=3.6",
    license="EUPL-1.2-or-later",
    install_requires=[
        "numpy>=1.25.0",
        "jax>=0.4.13",
        "jaxlib>=0.4.13",
        "scipy>=1.11.1",
        "scikit-learn>=1.3.0",
        "sympy>=1.12",
    ],
)
