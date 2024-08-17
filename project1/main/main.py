from variables_and_types import variables_and_types as variables
from collection_and_usage import collection_and_usage
from file_handling import file_handing
from numpy_usage import numpy_test
from misc import test_misc
from control_statements import control_statements
from class_and_object import class_and_object
from project.config.env import os_name
from project.config_utils import toLower
from project.logger import Logger
from project.model.project import Project

"""
This method is called when the program starts
Note: This method is called only once
"""


def startup():
    Logger.print(f"test")  # This line will add a log to console

    print(toLower("Test"))
    print(os_name)

    p = Project()
    p.name = "test project"

    print(p.name)

    variables()  # Run variable method from variables_and_types.py
    collection_and_usage()  # Run collection method from collection_and_usage.py
    control_statements()  # Run control statements method from control_statements.py
    class_and_object()  # Run class and object method from class_and_object.py
    test_misc()  # Run test_misc method from misc.py
    # input_txt = input("Enter a text: ")  # Take input from user
    # print(input_txt)
    file_handing()  # Run file handling method from file_handling.py
    numpy_test()  # Run numpy test method from numpy_usage.py


if __name__ == '__main__':
    startup()
