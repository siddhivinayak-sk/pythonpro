def fun_list0():
    squares = [1, 4, 9, 16, 25]
    squares_2 = [36, 49, 64, 81, 100]
    vowels = ['a', 'e', 'i', 'o', 'u']
    squares_3 = squares
    squares_3.append(35)
    squares_3[5] = squares_3[5] + 1
    print(squares)
    print(vowels)
    print(squares[1])
    print(squares[-1])
    print(squares[2:3])
    print(squares + squares_2)
    print(squares_3)
    squares_3.remove(36)
    print(squares_3)
    squares_3.reverse()
    print(squares_3)
    print(len(squares_3))
    

