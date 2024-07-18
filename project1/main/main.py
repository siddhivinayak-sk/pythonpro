from project.config.env import os_name
from project.config_utils import toLower
from project.logger import Logger
from project.model.project import Project


def startup():
    Logger.print(f"test")
    print(toLower("Test"))
    print(os_name)
    
    
    p = Project()
    p.name = "test project"
    
    print(p.name)


if __name__ == '__main__':
    startup()
