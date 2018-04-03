import numpy as np
import matplotlib.pyplot as plt
import time

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



def quicksort(x, low, high):
    if low < high:
        partition = create_partition(x, low, high)
        quicksort(x, low, partition-1)
        quicksort(x, partition+1, high)
    else:
        return

def create_partition(x, low, high):
    pivot = low
    for i in range(low+1, high+1):
        if x[i] <= x[low]:
            pivot += 1
            x[i], x[pivot] = x[pivot], x[i]
    x[pivot], x[low] = x[low], x[pivot]
    return pivot





def bubble_sort(x):
    pass


if __name__=='__main__':
    x = [3, 5, 7, 1, 8, 2, 8, 5, 8, 1]
    start_time = time.time()
    x = merge_sort(x)
    end_time = time.time() - start_time
    print('sorted list with merge sort: %s in : %s s' % (x, end_time))

    y = [3, 5, 7, 1, 8, 2, 8, 5, 8, 1]
    start_time = time.time()
    quicksort(y, 0, len(x)-1)
    end_time = time.time() - start_time
    print('sorted list with quicksort: %s in : %s s' % (y, end_time))
