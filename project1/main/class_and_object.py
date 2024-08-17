"""
Class and Object
- Python is an object-oriented programming language.
- Almost everything in Python is an object, with its properties and methods.
- A Class is like an object constructor, or a "blueprint" for creating objects.
- Support inheritance, polymorphism, encapsulation
"""


def class_and_object():
    my_class1 = MyClass1()
    print(my_class1.x)
    my_class1.x = 10
    print(my_class1.x)
    print(my_class1)  # prints type and object id as __str__ is not implemented
    del my_class1.x  # Delete x property from object
    print(f"Deleted x property: {my_class1.x}")  # prints 25 as x property is deleted
    del my_class1  # Delete object

    my_class3 = Person("John", 36)  # Create an object of Person with argument constructor, no argument constructor will fail
    print(my_class3.name)
    print(my_class3.age)
    print(my_class3)  # prints instance to string
    my_class3.print_my_class2()  # prints instance to string
    my_class3.print_my_class2_msg("Hello")  # prints instance to string

    my_class4 = WorkingPerson("John", 36)  # Create an object of WorkingPerson with argument constructor

    my_class5 = Employee("John", 36, 123)  # Create an object of Employee with argument constructor
    my_class5.welcome()  # prints instance to string

    for obj in [my_class3, my_class4, my_class5]:
        print(f"Object instanceof Person: {isinstance(obj, Person)}")
        print(obj.who_am_i())


class MyClass0:  # Empty class
    pass


class MyClass1:
    x = 5 * 5


class Person:  # Parent class
    def __init__(self, name, age):  # The __init__() function is called automatically every time the class is being used to create a new object.
        self.name = name  # self refers the instance of the class and init method is used as a constructor
        self.age = age

    def __str__(self):  # __str__ is a special method in Python that is called by the print() function to get the string representation of the object.
        return f"{self.name} {self.age}"

    def print_my_class2(self):  # No argument method, self is mandatory to refer the instance of the class, no need to pass it while calling. Name of variable could be anything
        print(f"{self.name} {self.age}")

    def print_my_class2_msg(self, msg):  # Argument method
        print(f"{msg}: {self.name} {self.age}")

    def who_am_i(self):
        return "Person"


class WorkingPerson(Person):  # Child class with empty body, parent __init__ method is called
    pass


class Employee(WorkingPerson):
    def __init__(self, name, age, emp_id):
        # Person.__init__(self, name, age)  # Call parent class constructor
        super().__init__(name, age)  # super() is used to call the parent class constructor
        self.emp_id = emp_id

    def welcome(self):
        print(f"Welcome {self.name} {self.age} {self.emp_id}")

    def who_am_i(self):
        return "Employee"
