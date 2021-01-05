import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tlinalg",
    version="0.0.1",
    author="Etienne Levecque",
    author_email="etienne.levecque.etu@univ-lille.fr",
    description="3rd order linear algebra functions such as product and factorization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Labaro/tlinalg",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=['numpy']
)