import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="stargazer",
    version="0.0.6",
    author="Pietro Battiston, Matthew Burke",
    author_email="me@pietrobattiston.it, matthew.wesley.burke@gmail.com",
    description="Nicely formatted regression reporting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/StatsReporting/stargazer",
    packages=setuptools.find_packages(),
    license='GPLv2',
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
)
