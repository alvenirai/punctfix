import setuptools


with open("./README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('./requirements.txt', encoding="utf-8") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="punctfix",
    version="0.0.4",
    author="Rasmus Arpe Fogh Egebæk",
    author_email="rasmus@alvenir.ai",
    description="Punctuation restoration library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=requirements,
    license_file="LICENCE.txt",
    url="https://github.com/danspeech/punctfix",
    classifiers=[
        "Programming Language :: Python :: 3",
        'Development Status :: 5 - Production/Stable',
        "Operating System :: OS Independent",
    ],
)
