import random

from project.model.project import Project

x = "This is a global variable"

"""
This method is created to test variables in python
"""


def variables_and_types():
    print("===========Variables and types===========")

    # Variables

    # Numbers
    a = int(100)
    b = float(100.0)
    i = complex(1, 2)
    i1 = 3 + 5j  # j as imaginary part

    # Text
    c = str("100")

    # Boolean
    d = bool(True)

    # Sequence
    e = list([1, 2, 3])
    f = tuple((1, 2, 3))
    n = range(10)

    # Mapping type
    g = dict({"a": 1, "b": 2})

    # Set type
    h = set({1, 2, 3})
    o = frozenset({1, 2, 3})

    # Binary type
    j = bytes(10)
    k = bytearray(10)
    l = memoryview(bytes(10))

    # None type
    m = None

    # Class object type
    p = Project()

    q, r, s = 'apple', 'banana', 'cherry'
    t = ['mango', 'date', 'kiwi']
    u, v, w = t  # Unpacking a list
    global x  # Mark this x is Global variable
    x = "global variable value changed now"

    print(x)  # This will print global x variable

    print(type(a))  # Print type of a

    print(ij := i + i1)  # Print complex number addition

    print(float(a))  # Convert int to float, force cast

    print(random.randrange(1, 10))  # Generate random number between 1 to 10

    # String type
    str1 = "Hello"
    str2 = 'World'
    str3 = "Pen's color is white"  # Single quote inside double quote
    str3 = """
            This is a
            multiline
            """
    str4 = '''
            This is a
            multiline
            '''
    str5 = str1[1]  # Get character at index 1, but there is no char type in python so it will be string only
    for x in str1:  # Loop through string
        print(x)
    str1_length = len(str1)  # Get length of string
    str6 = str1 + str2  # Concatenate two strings
    str7 = str1 * 3  # Repeat string 3 times
    str8 = "Hello" "World"  # Concatenate two strings without + operator
    str9 = "Hello" + "World"  # Concatenate two strings with + operator
    str_check = "Hello" in str1  # Check if "Hello" is in str1
    str_check_not = "Hello" not in str1  # Check if "Hello" is not in str1
    str10 = str1.upper()  # Convert string to uppercase
    str11 = str1.lower()  # Convert string to lowercase
    str12 = str1.strip()  # Remove whitespace from beginning or end
    str13 = str1.replace("H", "J")  # Replace H with J
    str14 = str1.split("e")  # Split string by e
    str15 = str1.capitalize()  # Capitalize first letter
    str16 = str1.casefold()  # Lowercase string
    str17 = str1.center(20)  # Center string in 20 characters
    str18 = str1.encode()  # Encode string
    str19 = str1.endswith("o")  # Check if string ends with o
    str20 = str1.expandtabs(2)  # Set tab size to 2
    str21 = str1.find("e")  # Find first occurrence of e
    str22 = str1.index("e")  # Find first occurrence of e
    str23 = str1.isalnum()  # Check if all characters are alphanumeric
    str24 = str1.isalpha()  # Check if all characters are alphabetic
    str25 = str1.isdecimal()  # Check if all characters are decimals
    str26 = str1.isdigit()  # Check if all characters are digits
    str27 = str1.isidentifier()  # Check if string is a valid identifier
    str28 = str1.islower()  # Check if all characters are lowercase
    str29 = str1.isnumeric()  # Check if all characters are numeric
    str30 = str1.isprintable()  # Check if all characters are printable
    str31 = str1.isspace()  # Check if all characters are whitespaces
    str32 = str1.istitle()  # Check if string is titlecased
    str33 = str1.isupper()  # Check if all characters are uppercase
    str34 = str1.join(str2)  # Join strings
    str35 = str1.ljust(20)  # Left justify string in 20 characters
    str36 = str1.lower()  # Lowercase string
    str37 = str1.lstrip()  # Remove whitespace from beginning
    str38 = str1.partition("e")  # Partition string by e
    str39 = str1.replace("H", "J")  # Replace H with J
    str40 = str1.rfind("e")  # Find last occurrence of e
    str41 = str1.rindex("e")  # Find last occurrence of e
    str42 = str1.rjust(20)  # Right justify string in 20 characters
    str43 = str1.rpartition("e")  # Partition string by e from right
    str44 = str1.rsplit("e")  # Split string by e from right
    str45 = str1.rstrip()  # Remove whitespace from end
    str46 = str1[1:3]  # Slice string
    str47 = str1[-3:-1]  # Slice string from right
    int1 = 10
    str48 = f"Hello {int1}"  # Format string
    str49 = f"Hello {int1:.2f}"  # Format string with 2 decimal places
    str50 = "welcome to \"xxx\" jungle"  # Escape double quote

    # Boolean
    bool1 = True
    bool2 = 2 > 1
    bool3 = bool("Hello")  # Convert string to boolean, most will return True except empty string
    bool4 = bool(False)  # Convert False to boolean
    bool5 = bool(None)  # Convert False to boolean
    bool6 = bool(0)  # Convert False to boolean
    bool7 = bool("")  # Convert False to boolean
    bool8 = bool(())  # Convert False to boolean
    bool9 = bool([])  # Convert False to boolean
    bool10 = bool({})  # Convert False to boolean
    print(bool1 := bool1)  # Same as print(bool1)

    return ''
