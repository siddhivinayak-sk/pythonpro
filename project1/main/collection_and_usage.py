from misc import do_nothing


def collection_and_usage():  # Collection and usage method: List, Tuple, Set, Dictionary
    print("===========Collection and usage===========")

    # List - preserve order, changeable, duplicate allowed, accessed by index
    # No built-in support for Arrays so list are used as arrays
    simple_list_1 = ['abc', 'def', 'ghi', 'jkl', 'mno']  # Create a list
    simple_list_2 = list(('abc', 'def', 'ghi', 'jkl', 'mno'))  # Same as above
    simple_list_3 = list()  # Create an empty list
    simple_list_3.append('pqr')  # Add an item to the list
    simple_list_3.insert(0, '000')  # Add an item at a specific index
    simple_list_3.remove('pqr')  # Remove an item from the list
    del simple_list_2[0]  # Remove an item from a specific index
    simple_list_3.pop()  # Remove the last item from the list
    simple_list_2.pop(1)  # Remove an item from a specific index
    simple_list_2.clear()  # Remove all items from the list
    del simple_list_2  # Delete the list
    simple_list_4 = simple_list_1.copy()  # Copy a list
    simple_list_4 = list(simple_list_1)  # Copy a list by list constructor
    simple_list_5 = simple_list_1  # Copy a list
    simple_list_6 = simple_list_1[1:3]  # Copy a list by slicing
    simple_list_7 = simple_list_1[-1]  # Copy a list
    simple_list_8 = simple_list_1[-3:-1]  # Copy a list
    simple_list_9 = simple_list_1[1:]  # Copy a list
    simple_list_10 = simple_list_1[:3]  # Copy a list
    str_ele_1 = simple_list_1[0]  # Access an item from the list
    simple_list_2 = ["abc"]
    simple_list_9 = simple_list_2.extend(["pqr", "stu"])  # Add multiple items to the list by extending with another list
    simple_list_9 = simple_list_1 + simple_list_2  # Add two list
    simple_list_10 = [x for x in
                      simple_list_1]  # Copy a list by list comprehension 'newlist = [expression for item in iterable if condition == True]'
    simple_list_11 = [x for x in simple_list_1 if "a" in x]  # Copy a list by list comprehension with condition
    simple_list_12 = simple_list_1.sort()  # Sort the list
    simple_list_13 = simple_list_1.sort(reverse=True)
    simple_list_14 = simple_list_1.sort(key=str.lower)  # Sort the list by key function
    simple_list_15 = simple_list_1.reverse()  # Reverse the list order

    # Loops
    for x in simple_list_1:  # Loop through the list
        do_nothing()
    for i in range(len(simple_list_1)):  # Loop through the list by index
        do_nothing()
    i = 0
    while i < len(simple_list_1):  # Loop through the list by index
        do_nothing()
        i += 1
    [do_nothing() for x in simple_list_1]  # List comprehension

    # Tuple - preserve order, unchangeable, duplicate allowed, accessed by index, can contain different data types, unpack
    simple_tuple_1 = ('abc', 'def', 'ghi', 'jkl', 'mno')  # Create a tuple
    simple_tuple_2 = tuple(('abc', 123, 100.02, True))  # Same as above with different data types
    simple_tuple_3 = tuple()  # Create an empty tuple
    str_ele_2 = simple_tuple_1[0]  # Access an item from the tuple
    simple_tuple_4 = simple_tuple_1 + simple_tuple_2  # Add two tuples
    simple_tuple_5 = simple_tuple_1 * 2  # Multiply a tuple
    simple_tuple_6 = tuple(simple_list_1)  # Convert a list to a tuple
    simple_tuple_7 = tuple(simple_tuple_1)  # Copy a tuple
    simple_tuple_8 = simple_tuple_1[1:3]  # Copy a tuple by slicing
    simple_tuple_9 = simple_tuple_1[-1]  # Copy a tuple
    simple_tuple_10 = simple_tuple_1[-3:-1]  # Copy a tuple
    simple_tuple_11 = simple_tuple_1[1:]  # Copy a tuple
    simple_tuple_12 = simple_tuple_1[:3]  # Copy a tuple
    list_by_tuple_1 = list(simple_tuple_1)  # Convert a tuple to a list
    list_by_tuple_1[0] = 'pqr'  # Modify value in a list and create a tuple with list
    simple_tuple_13 = tuple(list_by_tuple_1)  # Convert a list to a tuple
    (var1_str, var2_int, var3_float, var4_bool) = simple_tuple_2  # Unpack a tuple
    (var1_str, var2_str,
     *var3_str) = simple_tuple_1  # Unpack a tuple, when the number of variables is unknown use * before the variable name

    # Set - unordered, unindexed, no duplicate, can contain different data types, access element through loop
    simple_set_1 = {'abc', 'def', 'ghi', 'jkl', 'mno'}  # Create a set
    simple_set_2 = {'abc', 100, True, 100.02}  # Create a set with different data types
    simple_set_3 = set(('abc', 'def', 'ghi', 'jkl', 'mno'))  # Same as above
    simple_set_4 = set()  # Create an empty set
    simple_set_4.add('pqr')  # Add an item to the set
    simple_set_4.update(['stu', 'vwx'])  # Add multiple items to the set
    simple_set_4.remove('pqr')  # Remove an item from the set
    simple_set_4.discard('stu')  # Remove an item from the set
    simple_set_4.pop()  # Remove the last item from the set
    simple_set_4.clear()  # Remove all items from the set
    del simple_set_4  # Delete the set
    simple_set_5 = simple_set_1.copy()  # Copy a set
    simple_set_6 = set(simple_set_1)  # Copy a set by set constructor
    simple_set_7 = simple_set_1  # Copy a set
    simple_set_8 = simple_set_1.union(simple_set_2)  # Add two sets
    simple_set_9 = simple_set_1.intersection(simple_set_2)  # Get common items from two sets
    simple_set_10 = simple_set_1.difference(simple_set_2)  # Get different items from two sets
    simple_set_11 = simple_set_1.symmetric_difference(simple_set_2)  # Get items that are not present in both sets
    simple_set_12 = simple_set_1.isdisjoint(simple_set_2)  # Check if two sets have no common items
    simple_set_13 = simple_set_1.issubset(simple_set_2)  # Check if all items in the set are present in another set
    simple_set_14 = simple_set_1.issuperset(simple_set_2)  # Check if all items in another set are present in the set
    simple_set_15 = simple_set_1.symmetric_difference_update(simple_set_2)  # Update the set with items that are not present in both sets
    simple_set_16 = simple_set_1.update(simple_set_2)  # Add items from another set to the set
    simple_set_17 = simple_set_1.intersection_update(simple_set_2)  # Remove items that are not present in both sets
    simple_set_18 = simple_set_1.difference_update(simple_set_2)  # Remove items that are present in both sets
    simple_set_19 = simple_set_1.discard('abc')  # Remove an item from the set

    # Dictionary - key:value pair, ordered, changeable, no duplicate key, accessed by key, can contain different data types values
    simple_dict_1 = {'name': 'John', 'age': 36, 'city': 'New York'}  # Create a dictionary
    simple_dict_2 = dict(name='John', age=36, city='New York')  # Same as above by using dict constructor
    simple_dict_3 = dict([('name', 'John'), ('age', 36), ('city', 'New York')])  # Same as above by using list of tuples
    simple_dict_4 = dict(zip(['name', 'age', 'city'], ['John', 36, 'New York']))  # Same as above by using zip
    simple_dict_5 = dict({('name', 'John'), ('age', 36), ('city', 'New York')})  # Same as above by using set of tuples
    simple_dict_6 = dict()  # Create an empty dictionary
    simple_dict_6['name'] = 'John'  # Add an item to the dictionary
    simple_dict_6['age'] = 36  # Add an item to the dictionary
    simple_dict_6['city'] = 'New York'  # Add an item to the dictionary
    name_from_dict = simple_dict_1['name']  # Access an item from the dictionary
    simple_dict_7 = simple_dict_1.get('name')  # Access an item from the dictionary
    simple_dict_8_keys = simple_dict_1.keys()  # Get all keys from the dictionary
    simple_dict_9_values = simple_dict_1.values()  # Get all values from the dictionary
    simple_dict_10_items = simple_dict_1.items()  # Get all items from the dictionary
    simple_dict_11 = simple_dict_1.copy()  # Copy a dictionary
    simple_dict_12 = dict(simple_dict_1)  # Copy a dictionary by dict constructor
    simple_dict_13 = simple_dict_1  # Copy a dictionary
    simple_dict_14 = simple_dict_1.pop('name')  # Remove an item from the dictionary
    simple_dict_15 = simple_dict_1.popitem()  # Remove the last item from the dictionary
    simple_dict_16 = simple_dict_1.clear()  # Remove all items from the dictionary
    del simple_dict_1  # Delete the dictionary
    child1 = {
        "name": "Emil",
        "year": 2004
    }
    child2 = {
        "name": "Tobias",
        "year": 2007
    }
    child3 = {
        "name": "Linus",
        "year": 2011
    }
    my_family = {  # Nested dictionary
        "child1": child1,
        "child2": child2,
        "child3": child3
    }
    str_first_child_name = my_family["child1"]["name"]  # Access an item from the nested dictionary

    # Iterable and iterators
    # Iterable - object that can be iterated over like: list, tuple, set, dictionary
    # Iterator - class follows iterator protocol, has __iter__() and __next__() methods
    test_tuple = ("apple", "banana", "cherry")
    test_tuple_iterator = iter(test_tuple)
    print(next(test_tuple_iterator))
    print(next(test_tuple_iterator))
    print(next(test_tuple_iterator))

    return 0
