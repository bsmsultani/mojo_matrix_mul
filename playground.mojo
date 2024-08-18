from parallelised_matrix import Matrix
from time import sleep

import algorithm 

def main():

    print("Looping")
    # create a 3, 3 matrix with random numbers
    var A = Matrix[3, 3].rand()

    # loop and print its values
    for row in range(3):
        for col in range(3):
            var num = A.load[1](row, col)
            print(num)
    
    sleep(1)
    print("Loading all elements with SIMD")
    var res = A.data.load[width=6]()
    print(res)
