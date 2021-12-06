import setuptools

with open('README.md', 'r') as infile:
    long_description = infile.read()

setuptools.setup(
    name='asperities',
    version='1.0.1',
    author='Christer Dreierstad',
    author_email='christerdr@outlook.com',
    description='Package for characterizing voids/asperities in geometries',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/chdre/asperities',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=['scipy'],
    include_package_data=True,
)
