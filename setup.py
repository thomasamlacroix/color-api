"""
Setup program to install the API
"""
from setuptools import find_packages
from setuptools import setup


PR_NAME='color_rise_api'
VERSION='0.0.1'

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name=PR_NAME,
      version=VERSION,
      description="Color-rise API",
      license="MIT",
      author="TL",
      author_email="thomas.am.lacroix@gmail.com",
      url="https://github.com/thomasamlacroix/color-api",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="test",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
