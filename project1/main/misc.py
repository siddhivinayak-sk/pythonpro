import datetime
import math
import json
import re

"""
Misc feature tests
"""


def test_misc():
    # Date
    now1 = datetime.datetime.now()
    print(f"Current date with now: {now1}")
    print(f"Current date and time: {now1.strftime('%Y-%m-%d %H:%M:%S')}")
    then1 = datetime.datetime(2020, 5, 17)
    print(f"Then date with now: {then1}")

    # Math
    num1 = min(10, 2, 11, 1, 3)
    num2 = max(10, 2, 11, 1, 3)
    num3 = abs(-7.25)
    num4 = pow(4, 3)
    num5 = math.sqrt(64)
    num6 = math.ceil(1.4)
    num7 = math.floor(1.4)
    num8 = math.pi
    num9 = math.e

    # JSON
    json_str = '{"name": "John", "age": 30, "city": "New York"}'
    json_obj = json.loads(json_str)  # Convert JSON string to Python dictionary
    print(f"JSON object name: {json_obj['name']}")
    dict1 = {"name": "John", "age": 30, "city": "New York"}
    json_str = json.dumps(dict1)  # Convert Python dictionary to JSON string; it converts dict, list, tuple, string, int, float, True, False, None
    json_str = json.dumps(
        dict1,
        indent=4,
        separators=(". ", " = "),
        sort_keys=True,
        default=str,
        skipkeys=True,
        ensure_ascii=True,
        check_circular=True,
        allow_nan=True,
        cls=None,
    )
    print(json_str)

    # RegEx
    text1 = "This is a pen and this can be used to write essay. Its color is blue. It is a good pen. Its look is good."
    result1 = re.search("^This.*good.$", text1)  # Search the string to see if it starts with "This" and ends with "good."
    print(result1)
    result2 = re.findall("pen", text1)  # Return a list containing every occurrence of "pen"
    print(result2)
    result3 = re.split("pen", text1)  # Return a list where the string has been split at each match
    print(result3)
    result4 = re.sub("pen", "pencil", text1)  # Replace every occurrence of "pen" with "pencil"
    print(result4)
    result5 = re.match("This", text1)  # Search and result, if no match, return None
    print(result5)
    print(result5.span())
    print(result5.string)
    print(result5.group())


def do_nothing():
    return 0
