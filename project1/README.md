## PIP
PIP is a package manager for Python packages. You can use pip to install packages from the Python Package Index and other indexes.

```bash
pip --version
pip install SomePackage   # latest version
pip install SomePackage==1.0.4  # specific version
pip install SomePackage>=1.0.4  # minimum version
pip install SomePackage<=1.0.4  # maximum version
pip install SomePackage~=1.0.4  # compatible version
pip install SomePackage[PDF]  # install extra dependencies
pip install SomePackage --no-deps  # ignore dependencies
pip install SomePackage --no-cache-dir  # ignore cache
pip install SomePackage --no-binary :all:  # ignore binary
pip install SomePackage --no-build-isolation  # ignore build isolation
pip install SomePackage --no-use-pep517  # ignore PEP 517
pip install SomePackage --no-clean  # ignore clean
pip install SomePackage --no-compile  # ignore compile
pip install SomePackage --no-index  # ignore index
pip install SomePackage --no-deps  # ignore dependencies
pip install SomePackage --no-warn-script-location  # ignore script location
pip install SomePackage --no-warn-conflicts  # ignore conflicts
pip install SomePackage --no-warn-script-location  # ignore script location
pip install SomePackage --no-warn-conflicts  # ignore conflicts
pip uninstall SomePackage # uninstall package
pip list # list installed packages
pip show SomePackage # show package details
pip search SomePackage # search package
pip freeze > requirements.txt # save installed packages
pip install -r requirements.txt # install from requirements.txt
pip install --upgrade SomePackage # upgrade package
pip install --upgrade pip # upgrade pip
pip install --upgrade setuptools # upgrade setuptools
pip install --upgrade wheel # upgrade wheel
```

# Project1

URL: https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/


Local Install:
python -m pip install -e .

Uninstall:
python -m pip uninstall project1

Package:
python -m pip install build

Source Distro:
python -m build --sdist

Wheel:
python -m build --wheel
python -m pip install .\dist\project1-0.0.0-py3-none-any.whl


Check dist and upload on pypi:
twine check dist/*
twine upload dist/*



https://packaging.python.org/en/latest/glossary/#term-Python-Package-Index-PyPI
configure auth token with $HOME/.pypirc
[pypi]
username = __token__
password = <the token value, including the `pypi-` prefix>