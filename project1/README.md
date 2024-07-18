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