import io
import os
import re

from setuptools import find_packages
from setuptools import setup
from requirements import get_req

def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())


setup(
    name="siddhivinayak.sk.project1",
    version="0.0.0",
    url="https://www.siddhivinayaksk.co.in/",
    license='MIT',

    author="Sandeep Kumar",
    author_email="test@test.com",

    description="Test Project1",
    long_description="Test Project1",

    packages=find_packages(),
    install_requires=get_req(),

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    
    entry_points={ 
        "console_scripts": [
            "project1=main:main",
        ],
    },
    
)
