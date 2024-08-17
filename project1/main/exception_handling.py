"""
Support exception handling for the application.
try, except, else, finally are the main blocks of exception handling.
"""


def exception_handling():
    try:  # Code block that can raise an exception
        raise Exception("This is an exception")  # This line will raise an exception
    except NameError as e:  # This block will catch the exception, could be multiple
        print(f"NameError: {e}")
    except Exception as e:
        print(f"Exception: {e}")
    else:  # This block will execute if there is no exception
        print("No exception")
    finally:  # This block will execute always
        print("Finally block")
