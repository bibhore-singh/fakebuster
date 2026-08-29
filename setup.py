from setuptools import setup, find_packages

setup(
    name="fakebuster",
    version="1.0.0",
    description="AI-Powered Fake News Detection & Source Credibility Engine",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Bibhore Raj",
    author_email="bibhoreraj1@gmail.com",
    url="https://github.com/bibhore-singh/fakebuster",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=[
        "Flask>=2.2.0",
        "scikit-learn>=1.2.0",
        "pandas>=1.5.0",
        "numpy>=1.23.0",
    ],
)
