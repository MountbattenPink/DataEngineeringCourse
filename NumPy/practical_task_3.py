import numpy as np

#Practical Task 3: Array Manipulation with Separate Output Function in NumPy
# Objective: Develop a set of Python functions using NumPy to manipulate arrays through operations such as transposing, reshaping, splitting, and combining.

def print_array(arr, msg=""):
    print(msg, arr)


# Requirements:
# 1. Array Creation:
# o Generate a multi-dimensional NumPy array with random values. The array should have a complex structure
# (e.g., a 6x6 matrix of integers) to clearly demonstrate changes through each manipulation.
n = 6
data = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        data[i][j] = np.random.randint(1, 1000)

print_array(data, 'Generated matrix: ')



# 2. Array Manipulation Functions:
# o Transpose Function: Create a function to transpose the array and return the result.
def transpose(arr):
    return np.transpose(arr)

print_array(transpose(data), '\nTransposed matrix: ')
np.testing.assert_array_equal(data, transpose(transpose(data)))



# o Reshape Function: Develop a function to reshape the array into a new configuration (e.g., from a 6x6 matrix to a 3x12 matrix) and return the reshaped array.
def reshape(arr, i, j):
    return np.reshape(arr, (i, j))
print_array(reshape(data, 9, 4), '\nReshaped matrix: ')
np.testing.assert_array_equal(data, reshape(reshape(data, 9, 4), 6, 6))



# o Split Function: Implement a function that splits the array into multiple sub-arrays along a specified axis and returns the sub-arrays.
def split(array, indices_or_sections, axis):
    return np.split(array, indices_or_sections, axis=axis)

print_array(split(data, 3, 1),'\nSplitted matrix: ')



# o Combine Function: Construct a function that combines several arrays into one and returns the combined array.
def combine(arrays, axis):
    return np.concatenate(arrays, axis)

np.testing.assert_array_equal(combine(split(data, 3, 0), 0), data)
np.testing.assert_array_equal(combine(split(data, 3, 1), 1), data)

