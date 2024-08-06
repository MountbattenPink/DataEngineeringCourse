# Practical Task 4: Comprehensive Data Handling and Analysis with NumPy
# Objective: Develop a set of Python functions using NumPy to handle reading/writing data and performing aggregate analyses on arrays.

import numpy as np
import practical_task_3 as task3
def print_array(arr, msg=""):
    print(msg, arr)


# Requirements:
# 1. Array Creation:
# o Generate a multi-dimensional NumPy array with random values. The array should have a complex structure (e.g., a 10x10 matrix of integers)
# to clearly demonstrate changes through each manipulation.

n = 10
data = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        data[i][j] = np.random.randint(1, 1000)

print_array(data, 'Generated matrix: ')


# 2. Data I/O Functions:
# o Save Function: Create a function to save the array to a text file, a CSV file, and a binary format (.npy or .npz).
def save(arr):
    np.savetxt("output.txt", arr, fmt='%s')
    np.savetxt("output.csv", arr, delimiter=',', fmt='%s')
    np.save("output.npy", arr)
    print("Data saved")

save(data)


# o Load Function: Develop a function to load the array from each of the saved formats back into NumPy arrays.
def load():
    txt_loaded = np.loadtxt("output.txt")
    print_array(txt_loaded, "\nLoaded from txt")
    csv_loaded = np.loadtxt("output.csv", delimiter=',')
    print_array(csv_loaded, "\nLoaded from csv")
    binary_loaded = np.load("output.npy")
    print_array(binary_loaded, "\nLoaded from binary")
    return txt_loaded, csv_loaded, binary_loaded

loaded_arrays = load()

np.testing.assert_array_equal(loaded_arrays[0], data)
np.testing.assert_array_equal(loaded_arrays[1], data)
np.testing.assert_array_equal(loaded_arrays[2], data)

# 3. Aggregate Functions:
# o Summation Function: Create a function to compute the summation of all elements.
def get_summation_numpy(arr):
    return np.sum(arr)

def get_summation_python(arr):
    sum = 0
    for i in range(n):
        for j in range(n):
            sum += arr[i][j]
    return sum

np.testing.assert_equal(get_summation_numpy(data), get_summation_python(data))
np.testing.assert_equal(get_summation_numpy(data), get_summation_numpy(data.reshape(1, 100)))
np.testing.assert_equal(get_summation_numpy(data), get_summation_numpy(data.reshape(1, 100)))

# o Mean Function: Develop a function to calculate the mean of the array.
def get_mean(arr):
    return np.mean(arr)

np.testing.assert_equal(get_mean(data), get_mean(data.reshape(1, 100)))
np.testing.assert_equal(get_mean(data), get_mean(task3.transpose(data)))


# o Median Function: Implement a function to find the median of the array.
def get_median(arr):
    return np.median(arr)

np.testing.assert_equal(get_median(data), get_median(data.reshape(1, 100)))
np.testing.assert_equal(get_median(data), get_median(task3.transpose(data)))


# o Standard Deviation Function: Construct a function to calculate the standard deviation of the array.
def get_standard_deviation(arr, axis, ddof):
    return np.std(arr, axis=axis, ddof=ddof)

print_array(get_standard_deviation(data, 0, 0), "\nStandard Deviation 0 axis: ")
print_array(get_standard_deviation(data, 1, 0), "\nStandard Deviation 1 axis: ")



# o Axis-Based Aggregate Functions: Create functions to apply these aggregate operations along different axes (row-wise and column-wise).
def get_summation_numpy_axis(arr, axis):
    return np.sum(arr, axis)

print_array(get_summation_numpy_axis(data, 0), "\nGet summation with axis 0: ")
print_array(get_summation_numpy_axis(data, 1), "\nGet summation with axis 1: ")
np.testing.assert_array_equal(get_summation_numpy_axis(data, 0), get_summation_numpy_axis(task3.transpose(data), 1))
np.testing.assert_array_equal(get_summation_numpy_axis(data, 1), get_summation_numpy_axis(task3.transpose(data), 0))


def get_mean_numpy_axis(arr, axis):
    return np.mean(arr, axis)

print_array(get_mean_numpy_axis(data, 0), "\nGet mean with axis 0: ")
print_array(get_mean_numpy_axis(data, 1), "\nGet mean with axis 1: ")
np.testing.assert_array_equal(get_mean_numpy_axis(data, 0), get_mean_numpy_axis(task3.transpose(data), 1))
np.testing.assert_array_equal(get_mean_numpy_axis(data, 1), get_mean_numpy_axis(task3.transpose(data), 0))


def get_median_numpy_axis(arr, axis):
    return np.median(arr, axis)

print_array(get_median_numpy_axis(data, 0), "\nGet median with axis 0: ")
print_array(get_median_numpy_axis(data, 1), "\nGet median with axis 1: ")
np.testing.assert_array_equal(get_median_numpy_axis(data, 0), get_median_numpy_axis(task3.transpose(data), 1))
np.testing.assert_array_equal(get_median_numpy_axis(data, 1), get_median_numpy_axis(task3.transpose(data), 0))


# o Loading and Verification: Load the arrays back and verify their integrity using the
# load function and the print_array function to display the original and loaded arrays.
print_array(data, "\nData array after all the manipulations: ")
print_array(load()[0], "\nLoaded from txt: ")
np.testing.assert_array_equal(data, load()[0])