from setuptools import setup, find_packages

setup(
    name="InfoGAN",
    version="0.1.0",
    description="A deep learning project that is build for GAN for the MNIST digit dataset",
    author="Atikul Islam Sajib",
    author_email="atikul.sajib@ptb.de",
    url="https://github.com/atikul-islam-sajib/InfoGAN.git",  # Update with your project's GitHub repository URL
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="WGAN machine-learning",
    project_urls={
        "Bug Tracker": "https://github.com/atikul-islam-sajib/InfoGAN.git/issues",
        "Documentation": "https://atikul-islam-sajib.github.io/InfoGAN-deploy/",
        "Source Code": "https://github.com/atikul-islam-sajib/InfoGAN.git",
    },
)
