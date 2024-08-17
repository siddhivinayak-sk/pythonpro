from misc import do_nothing

"""
Control statements
 - Method
 - If-else
 - For loop
 - While loop
"""

my_var1 = -1  # Global variable


def control_statements():
    print("===========Control Statements===========")

    # If-else
    a = 10
    b = 20
    if a > b:  # If statement
        print("a is greater than b")
    elif a == b:
        print("a is equal to b")
    else:
        print("b is greater than a")
    if a > b: print("a is greater than b")  # One line if statement
    print("A") if a > b else print("B")  # One line if-else statement
    print("A") if a > b else print("=") if a == b else print("B")  # One line if-elif-else statement
    x = 41
    if x > 10:
        print("Above ten,")
        if x > 20:  # Nested if
            print("and also above 20!")
        else:
            print("but not above 20.")
    if x != 100:
        pass  # Use pass statement to avoid getting an error when empty block

    # While loop
    i = 1
    while i < 10:  # While loop
        i += 1
        if i == 5:
            break  # Break loop statement when condition mate
        if i == 3:
            continue  # Continue loop statement when condition mate
        print(i)
    else:
        print("loop condition is false")  # Else block will run when loop condition is false

    # For loop
    for num in range(1, 30, 3):  # For loop, increment by 3 from 1 to 30
        if num == 27:  # Break loop statement when condition mate
            break
        if num == 6:  # Continue loop statement when condition mate
            continue
        print(num)
    else:
        print("loop condition is false")  # Else block will run when loop condition is false

    for letter in 'Python':
        pass  # Use pass statement to avoid getting an error when empty block

    # function calling
    do_nothing()
    my_function()
    my_function_with_argument(1, 2)
    my_function_with_argument(arg1=1, arg2=2)  # Passing arguments by name
    temp1 = my_function_with_return_value(1, 2)
    my_function_with_arbitrary_argument(1, 2, 3, 4, 5)  # Passing arbitrary arguments
    my_function_with_arbitrary_argument_and_name(1, size=1)  # Passing arbitrary arguments with name
    my_function_with_default_argument(1)  # Passing default argument
    my_function_with_positional_args(1)  # Passing positional only argument

    # Lambda function (Anonymous function)
    l_function_1 = lambda arg1: arg1 + 10  # Lambda function with one argument
    l_function_2 = lambda arg1, arg2: arg1 + arg2 + 10  # Lambda function with two arguments

    print(l_function_1(5))
    l_function_4 = l_function_3(2)
    print(l_function_4(5))  # Nested lambda function usage


def my_function():  # method by default returns None
    my_var1 = 1  # Local variable
    print("Hello from a function" + str(my_var1))


def my_function_with_argument(arg1, arg2):
    print(f"Hello from a function arg1 = {arg1}, arg2 = {arg2}")


def my_function_with_return_value(arg1, arg2):  # Function with return value
    return arg1 + arg2


def my_function_with_arbitrary_argument(arg1, *args):  # Arbitrary arguments when number of arguments are unknown
    return arg1 + args[0]


def my_function_with_arbitrary_argument_and_name(arg1, **args):  # Arbitrary arguments when number of arguments are unknown with name
    return arg1 + args['size']


def my_function_with_default_argument(arg1, arg2=2):  # Default argument
    return arg1 + arg2


def my_function_with_no_body():
    pass  # Use pass statement to avoid getting an error when empty block


def my_function_with_positional_args(arg1, /):  # Positional only arguments need to add ', /' after arguments
    pass


def my_function_with_keyword_args(*, arg1, arg2):  # Keyword only arguments need to add ', *' before arguments
    pass


def l_function_3(arg1):
    return lambda arg2: arg2 * arg1  # Lambda function with nested lambda function


def my_function_outer():  # Support nested function
    x = "Jane"
    my_var1 = 2  # Local variable

    def my_function_inner():
        global my_var1  # Use global keyword to access global variable
        nonlocal x  # Use nonlocal keyword to access outer function variable
        x = "hello" + str(my_var1)

    my_function_inner()
    return x
