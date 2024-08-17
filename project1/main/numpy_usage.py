from numpy import random
import numpy as np

"""
NumPy is a Python library used for working with arrays.
It also has functions for working in domain of linear algebra, fourier transform, and matrices.
Numpy is faster than list in python, written in paython and c++.
It creates an array called ndarray, any number of dimensions.

Data Types:
i - integer
b - boolean
u - unsigned integer
f - float
c - complex float
m - timedelta
M - datetime
O - object
S - string
U - unicode string
V - fixed chunk of memory for other type ( void )
"""


def numpy_test():
    arr_zero_dimension = np.array(100)
    arr_one_dimension = np.array([1, 2, 3, 4, 5])
    arr_two_dimension = np.array([
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5]
    ])
    arr_three_dimension = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    arr_five_dimension = np.array([1, 2, 3], ndmin=5)
    print(arr_zero_dimension.ndim)
    print(arr_one_dimension.ndim)
    print(arr_two_dimension.ndim)
    print(arr_three_dimension.ndim)
    print(arr_five_dimension.ndim)
    print(arr_three_dimension[0, 1, 2])  # Access the element on index 2 (3rd element) on 1st index (2nd element) of 0th index (1st element)
    new_arr1 = arr_one_dimension[1:4]  # Slice elements from index 1 to index 4 (not included), slicing format is: [start:end:step]
    new_arr2 = arr_one_dimension[-3:-1]  # Negative meas counting from the end of the array
    new_arr3 = arr_one_dimension[::2]  # Return every other element from the entire array
    arr_str = np.array(["a", "b", "c", "e", "f"], dtype="S")
    print(arr_str.dtype)
    float_arr = np.array([1.1, 2.1, 3.1])
    int_arr = float_arr.astype(int)  # Convert float array to int array
    arr1 = np.array([1, 2, 3])
    arr2 = arr1.copy()  # Copy array, new array will be created
    arr3 = arr1.view()  # View array, no new array will be created, if there is change in source array, it will reflect in view array and vice versa
    print(arr2.base)  # copy returns None
    print(arr3.base)  # returns original data
    print(arr_two_dimension.shape)  # returns shape of array (index having number of elements in array)
    one_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    two_array = one_array.reshape(4, 3)  # reshape array to 4x3
    again_flat_array = two_array.reshape(-1)  # reshape array to 1D array
    for x in arr_three_dimension:  # Loop through 3D array
        for y in x:
            for z in y:
                print(z)
    for x in np.nditer(arr_three_dimension):  # Loop through 3D array using nditer
        print(f"x: {x}")
    for x in np.nditer(arr_three_dimension, flags=['buffered'], op_dtypes=['S']):  # Loop through 3D array using nditer with flags and op_dtypes
        print(f"y: {x}")
    for idx, x in np.ndenumerate(arr_three_dimension):  # Loop through 3D array using ndenumerate
        print(idx, x)
    #  np.concatenate((arr1, arr2), axis=0)  # Concatenate two arrays, axis=0 means concatenate along rows, axis=1 means concatenate along columns
    #  np.stack((arr1, arr2), axis=0)  # Stack two arrays, axis=0 means stack along rows, axis=1 means stack along columns
    #  np.hstack((arr1, arr2))  # Stack arrays horizontally
    #  np.vstack((arr1, arr2))  # Stack arrays vertically
    #  np.dstack((arr1, arr2))  # Stack arrays depth wise
    #  np.array_split(arr, 3)  # Split array into 3 sub-arrays
    #  np.hsplit(arr, 3)  # Split array into 3 sub-arrays horizontally
    #  np.where(arr > 4)  # Return indexes where condition is true
    #  np.searchsorted(arr, 4)  # Return index where element should be inserted to maintain order
    #  np.sort(arr)  # Sort array
    #  np.filter(arr, arr > 4)  # Filter array

    # Numpy Random
    random_int1 = random.randint(100)  # Generate random integer 0-100
    random_int_arr1 = random.randint(100, size=5)  # Generate random integer 0-100 array of size 5
    random_int_arr2 = random.randint(100, size=(3, 4))  # Generate random integer 0-100 2D array of size 12
    random_float1 = random.rand()  # Generate random float 0-1
    random_float_arr1 = random.rand(5)  # Generate random float 0-1 array of size 5
    random_float_arr1 = random.rand(2, 5)  # Generate random float 0-1 2D array of size 10
    rand_choice1 = random.choice([3, 5, 7, 9])  # Generate random choice from array
    rand_choice_arr2 = random.choice([3, 5, 7, 9], size=(3, 4))  # Generate random choice from array of size 12
    rand_choice_arr3 = random.choice([3, 5, 7, 9], size=(3, 4), p=[0.1, 0.3, 0.6, 0])  # Generate random choice from array of size 12 with probability
    random.shuffle(arr_one_dimension)  # Shuffle array
    permutation_arr1 = random.permutation(arr_one_dimension)  # Permute array

    #  Distributions - check the seaborn and distribution usage

    #  Universal Functions
    #  Converting iterative statements into a vector based operation is called vectorization.
    #  ufuncs are used to implement vectorization in NumPy which is way faster than iterating over elements.
    #  They also provide broadcasting and additional methods like reduce, accumulate etc. that are very helpful for computation.
    #  where boolean array or condition defining where the operations should take place.
    #  dtype defining the return type of elements.
    #  out output array where the return value should be copied.
    list1 = [1, 2, 3, 4]
    list2 = [4, 5, 6, 7]
    list3 = np.frompyfunc(local_add, 2, 1)(list1, list2)  # Add two arrays using frompyfunc where python function is passed
    list3 = np.add(list1, list2)  # Add two arrays
    list4 = np.subtract(list1, list2)  # Subtract two arrays
    list5 = np.multiply(list1, list2)  # Multiply two arrays
    list6 = np.divide(list1, list2)  # Divide two arrays
    list7 = np.power(list1, list2)  # Power of two arrays
    list8 = np.mod(list1, list2)  # Modulus of two arrays
    list9 = np.remainder(list1, list2)  # Remainder of two arrays
    list10 = np.divmod(list1, list2)  # Division and modulus of two arrays
    list11 = np.absolute(list1)  # Absolute of array
    list12 = np.around(list1)  # Round array
    list13 = np.floor(list1)  # Floor of array
    list13 = np.fix(list1)  # Fix of array (same as truc)
    list14 = np.ceil(list1)  # Ceil of array
    list15 = np.trunc(list1)  # Truncate of array
    list16 = np.exp(list1)  # Exponential of array
    list17 = np.log(list1)  # Logarithm of array
    list18 = np.log2(list1)  # Log base 2 of array
    list19 = np.log10(list1)  # Log base 10 of array
    list20 = np.sqrt(list1)  # Square root of array
    list21 = np.sin(list1)  # Sine of array
    list22 = np.cos(list1)  # Cosine of array
    list23 = np.tan(list1)  # Tangent of array
    list27 = np.hypot(list1, list2)  # Hypotenuse of array
    list28 = np.degrees(list1)  # Degrees of array
    list29 = np.radians(list1)  # Radians of array
    list30 = np.deg2rad(list1)  # Degrees to radians of array
    list31 = np.rad2deg(list1)  # Radians to degrees of array
    list32 = np.sinh(list1)  # Hyperbolic sine of array
    list33 = np.cosh(list1)  # Hyperbolic cosine of array
    list34 = np.tanh(list1)  # Hyperbolic tangent of array
    list35 = np.arcsinh(list1)  # Inverse hyperbolic sine of array
    list36 = np.arccosh(list1)  # Inverse hyperbolic cosine of array
    list38 = np.floor_divide(list1, list2)  # Floor division of array
    list39 = np.copysign(list1, list2)  # Copy sign of array
    list40 = np.greater(list1, list2)  # Greater than of array
    list41 = np.greater_equal(list1, list2)  # Greater than or equal of array
    list42 = np.less(list1, list2)  # Less than of array
    list43 = np.less_equal(list1, list2)  # Less than or equal of array
    list44 = np.equal(list1, list2)  # Equal of array
    list45 = np.not_equal(list1, list2)  # Not equal of array
    list46 = np.logical_and(list1, list2)  # Logical and of array
    list47 = np.logical_or(list1, list2)  # Logical or of array
    list48 = np.logical_xor(list1, list2)  # Logical xor of array
    list49 = np.logical_not(list1)  # Logical not of array
    list50 = np.maximum(list1, list2)  # Maximum of array
    list51 = np.minimum(list1, list2)  # Minimum of array
    list52 = np.fmax(list1, list2)  # Maximum of array
    list53 = np.fmin(list1, list2)  # Minimum of array
    list54 = np.modf(list1)  # Modf of array
    list55 = np.isreal(list1)  # Is real of array
    list56 = np.iscomplex(list1)  # Is complex of array
    list57 = np.isfinite(list1)  # Is finite of array
    list58 = np.isinf(list1)  # Is infinite of array
    list59 = np.isnan(list1)  # Is nan of array
    list60 = np.signbit(list1)  # Sign bit of array
    list61 = np.copysign(list1, list2)  # Copy sign of array
    list62 = np.nextafter(list1, list2)  # Next after of array
    list63 = np.spacing(list1)  # Spacing of array
    list64 = np.modf(list1)  # Modf of array
    list65 = np.ldexp(list1, list2)  # Ldexp of array
    list66 = np.frexp(list1)  # Frexp of array
    list67 = np.fabs(list1)  # Absolute of array
    list68 = np.add(list1, list2)  # Add of array
    list69 = np.reciprocal(list1)  # Reciprocal of array
    list70 = np.negative(list1)  # Negative of array
    list71 = np.positive(list1)  # Positive of array
    list72 = np.negative(list1)  # Negative of array
    sum1 = np.sum([list1, list2])  # Sum of array
    sum2 = np.cumsum([list1, list2])  # Cumulative sum of array
    prod1 = np.prod(list1)  # Product of array
    prod2 = np.cumprod(list1)  # Cumulative product of array
    diff1 = np.diff(list1)  # Difference of array
    diff2 = np.ediff1d(list1)  # Difference of array
    diff3 = np.gradient(list1)  # Gradient of array
    diff5 = np.trapz(list1)  # Trapezoidal of array
    lcm1 = np.lcm(4, 6)  # LCM of array
    lcm2 = np.lcm.reduce(list1)  # LCM of array
    hcf_gcd = np.gcd(4, 6)  # GCD (Greatest Common Denominator) of array
    hcf_gcd2 = np.gcd.reduce(list1)  # GCD (Greatest Common Denominator) of array
    unique = np.unique(list1)  # Unique of array
    intersect1d = np.intersect1d(list1, list2)  # Intersection of array
    union1d = np.union1d(list1, list2)  # Union of array
    setdiff1d = np.setdiff1d(list1, list2)  # Set difference of array
    setxor1d = np.setxor1d(list1, list2)  # Set exclusive-or of array
    pass


def local_add(x, y):
    return x + y