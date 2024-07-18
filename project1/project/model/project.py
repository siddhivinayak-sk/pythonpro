

class Project:
    def __init__(self):
        self.name = ""

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def toLowerCase(project_name):
        return project_name.lower()
