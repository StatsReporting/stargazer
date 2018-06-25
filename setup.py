import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="stargazer",
    version="0.0.1",
    author="Matthew Burke",
    author_email="matthew.wesley.burke@gmail.com",
    description="Nicely formatted regression reporting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mwburke/stargazer",
    packages=setuptools.find_packages(),
    license='GPLv2',
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
)
