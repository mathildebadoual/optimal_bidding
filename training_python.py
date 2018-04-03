import numpy as np
import matplotlib.pyplot as plt

""" Sorts Algorithms and Convexity """

def merge_sort(x):
    if len(x) <= 1:
        return x

    middle_index = int(len(x)/2)
    left = x[:middle_index]
    right = x[middle_index:]

    left = merge_sort(left)
    right = merge_sort(right)

    return merge(left, right)


def merge(left, right):
    result = []

    while not len(left) == 0 and not len(right) == 0:
        if left[0] < right[0] :
            result.append(left[0])
            left.pop(0)
        else:
            result.append(right[0])
            right.pop(0)

    while not len(left) == 0:
        result.append(left[0])
        left.pop(0)
    while not len(right) == 0:
        result.append(right[0])
        right.pop(0)

    return result



def quicksort(x):
    pass


def bubble_sort(x):
    pass


if __name__=='__main__':
    x = [3, 5, 7, 1, 8, 2, 8, 5, 8, 1]
    print(merge_sort(x))
