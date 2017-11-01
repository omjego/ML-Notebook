import numpy as np
import csv


def first():
    with open('winequality-red.csv', 'r') as f:
        wines = list(csv.reader(f, delimiter=';'))
    wines = np.array(wines[1:], dtype=float)
    # print wines.shape
    # creating array initialized to zero
    empty_arr = np.zeros((2, 3))
    print 'empty_arr : \n', empty_arr

    # creating array with random numbers
    random_arr = np.random.rand(3, 4)
    scaled = random_arr * 100
    scaled = scaled.astype(int)
    print 'random array : \n', scaled


def slicing():
    wines = np.genfromtxt('winequality-red.csv', delimiter=';', skip_header=1)
    print wines.dtype
    print wines[2, 3]

    sliced = wines[0:3, 3]
    sliced = np.array(sliced, dtype=float) / max(sliced)
    print sliced

    whole_col = wines[:, 3]
    print whole_col


def data_types():
    wines = np.genfromtxt('winequality-red.csv', delimiter=';', skip_header=1)
    print wines.dtype
    print wines.dtype.name
    wines = wines.astype('int')
    print wines.dtype.name

    wines = wines.astype('object')
    print wines[1, :]


def math_operations():
    # while reshaping total size of array must remain same.
    arr = np.arange(4)
    reshaped = arr.reshape(2, 2)
    # To flatten the array to 1D , use ravel
    print reshaped.ravel()
    wines = np.genfromtxt('winequality-red.csv', delimiter=';', skip_header=1)
    one_arr = (np.random.random(12)*100).astype('int')
    print one_arr
    print wines.shape
    print wines + one_arr
    third_wine = np.array(wines[3,])

    print wines
    # print third_wine.__class__
    # print third_wine
    # fourth_wine = wines[4,]

    # print fourth_wine
    # print (third_wine * fourth_wine)


def comparisons():
    wines = np.genfromtxt('winequality-red.csv', skip_header=1, delimiter=';')
    high_quality = wines[:, 11] > 7
    filtered = wines[high_quality, :]
    filtered = filtered[0:3, :]
    print filtered

    very_high = (wines[:, 11] > 5) & (wines[:, 10] > 7)
    filtered = wines[very_high, :]
    filtered = filtered[0 : 10, :]


if __name__ == '__main__':
    data_types()
