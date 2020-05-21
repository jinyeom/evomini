import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="evomini",
  version="0.1.0",
  author="Jin Yeom",
  author_email="jinyeom95@gmail.com",
  description="Minimal implementation of Neuroevolution",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/jinyeom/evomini",
  packages=setuptools.find_packages(),
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],
  python_requires='>=3.6',
)