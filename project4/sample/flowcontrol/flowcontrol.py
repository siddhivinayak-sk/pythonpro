from enum import Enum


def flow_control_if():
    # if statement
    x = int(input("Please enter an integer: "))
    if x < 0:
        x = 0
        print('Negative changed to zero')
    elif x == 0:
        print('Zero')
    elif x == 1:
        print('Single')
    else:
        print('More')


def flow_control_for1():
    words = ['cat', 'window', 'defenestrate']
    for w in words:
        print(w, len(w))


def flow_control_for2():
    # Create a sample collection
    users = {'Hans': 'active', 'Éléonore': 'inactive', '景太郎': 'active'}

    # Strategy:  Iterate over a copy
    for user, status in users.copy().items():
        if status == 'inactive':
            del users[user]

    # Strategy:  Create a new collection
    active_users = {}
    for user, status in users.items():
        if status == 'active':
            active_users[user] = status
    print(users)
    print(active_users)


def flow_control_for3():
    for n in range(2, 10):
        for x in range(2, n):
            if n % x == 0:
                print(n, 'equals', x, '*', n // x)
                break
        else:
            # loop fell through without finding a factor
            print(n, 'is a prime number')


def flow_control_for4():
    for num in range(2, 10):
        if num % 2 == 0:
            print("Found an even number", num)
            continue
        print("Found an odd number", num)


def flow_control_range1():
    for i in range(5):
        print(i)


def flow_control_range2():
    print(list(range(5, 10)))
    print(list(range(0, 10, 3)))
    print(list(range(-10, -100, -30)))


def flow_control_range3():
    a = ['Mary', 'had', 'a', 'little', 'lamb']
    for i in range(len(a)):
        print(i, a[i])


def flow_control_range4():
    a = sum(range(4))  # 0 + 1 + 2 + 3
    print(a)


def flow_control_pass():  # pass is simple keyword used where statement required but there is no statement
    while True:
        pass

    class MyEmptyClass:
        pass

    def initLog(*args):
        pass


def flow_control_match1(status):
    match status:
        case 400:
            return "Bad request"
        case 401 | 403 | 404:
            return "Not allowed"
        case 404:
            return "Not found"
        case 418:
            return "I'm a teapot"
        case _:
            return "Something's wrong with the internet"


def flow_control_match1_with_tuple(point):
    match point:
        case (0, 0):
            print("Origin")
        case (0, y):
            print(f"Y={y}")
        case (x, 0):
            print(f"X={x}")
        case (x, y):
            print(f"X={x}, Y={y}")
        case _:
            raise ValueError("Not a point")


class Point:
    x: int
    y: int


def flow_control_match1_with_point(point):
    match point:
        case Point(x=0, y=0):
            print("Origin")
        case Point(x=0, y=y):
            print(f"Y={y}")
        case Point(x=x, y=0):
            print(f"X={x}")
        case Point():
            print("Somewhere else")
        case _:
            print("Not a point")


def flow_control_match1_with_point_list(points):
    match points:
        case []:
            print("No points")
        case [Point(0, 0)]:
            print("The origin")
        case [Point(x, y)]:
            print(f"Single point {x}, {y}")
        case [Point(0, y1), Point(0, y2)]:
            print(f"Two on the Y axis at {y1}, {y2}")
        case _:
            print("Something else")


def flow_control_match1_with_point_and_if(point):
    match point:
        case Point(x, y) if x == y:
            print(f"Y=X at {x}")
        case Point(x, y):
            print(f"Not on the diagonal")


class Color(Enum):
    RED = 'red'
    GREEN = 'green'
    BLUE = 'blue'


def flow_control_match1_with_enum():
    color = Color(input("Enter your choice of 'red', 'blue' or 'green': "))
    match color:
        case Color.RED:
            print("I see red!")
        case Color.GREEN:
            print("Grass is green")
        case Color.BLUE:
            print("I'm feeling the blues :(")

