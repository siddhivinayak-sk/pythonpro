def fun_fib(n):  # write Fibonacci series up to n
    """Print a Fibonacci series up to n."""
    a, b = 0, 1
    while a < n:
        print(a, end=' ')
        a, b = b, a + b
    print()


def fun_fib3(n):
    """Print a Fibonacci series up to n."""
    result = []
    a, b = 0, 1
    while a < n:
        result.append(a)
        a, b = b, a + b
    print()


def ask_ok(prompt, retries=4, reminder='Please try again!'):  # default values to function argument
    while True:
        ok = input(prompt)
        if ok in ('y', 'ye', 'yes'):
            return True
        if ok in ('n', 'no', 'nop', 'nope'):
            return False
        retries = retries - 1
        if retries < 0:
            raise ValueError('invalid user response')
        print(reminder)


# Now call the function we just defined:
# fun_fib(2000)

fun_fib2 = fun_fib  # assign function to variable
fun_fib2(5)  # function call with function variable
print(fun_fib2)  # print the function location
print(fun_fib2(0))  # if no return then function returns None
fun_fib2(100)  # call function directly with file
fun_fib4 = fun_fib3(100)  # function call with argument and result is assigned to variable
fun_fib4  # function call result with assigned variable

# result = ask_ok('Enter your thought')
# print(f'you have entered: {result}')


"""
Default function argument value with variable
"""
i = 5


def fx1(arg=i):
    print(arg)


i = 6
fx1()


# Default argument value as mutable list
def fx2(a, L=[]):
    L.append(a)
    return L


print(fx2(1))
print(fx2(2))
print(fx2(3))


# Default argument value as not shared in subsequent calls
def fx3(a, L=None):
    if L is None:
        L = []
    L.append(a)
    return L


print(fx3(1))
print(fx3(2))
print(fx3(3))


# keyword argument
def fun_x4(test="xxx"):
    print(test)


fun_x4()
fun_x4("this is test line")


# argument with list and dictionary
def cheeseshop(kind, *arguments, **keywords):
    print("-- Do you have any", kind, "?")
    print("-- I'm sorry, we're all out of", kind)
    for arg in arguments:
        print(arg)
    print("-" * 40)
    for kw in keywords:
        print(kw, ":", keywords[kw])


cheeseshop("Limburger", "It's very runny, sir.",
           "It's really very, VERY runny, sir.",
           shopkeeper="Michael Palin",
           client="John Cleese",
           sketch="Cheese Shop Sketch")


# def f(pos1, pos2, /, pos_or_kwd, *, kwd1, kwd2):
#     -----------    ----------     ----------
#     |             |                  |
#     |        Positional or keyword   |
#     |                                - Keyword only
#     -- Positional only


# varargs
def concat(*args, sep="/"):
    return sep.join(args)


concat("earth", "mars", "venus")
concat("earth", "mars", "venus", sep=".")


# lambda
def make_incrementor(n):
    return lambda x: x + n


fx4 = make_incrementor(42)
print(fx4(0))
print(fx4(1))


# documenting functions
def my_function():
    """Do nothing, but document it.

    No, really, it doesn't do anything.
    """
    pass


print(my_function.__doc__)


# function annotation
def fx5(ham: str, eggs: str = 'eggs') -> str:
    print("Annotations:", fx5.__annotations__)
    print("Arguments:", ham, eggs)
    return ham + ' and ' + eggs


fx5('spam')

