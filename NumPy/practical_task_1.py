import numpy as np

# Practical Task 1: Basic Array Creation and Manipulation with NumPy

#Output Function: Implement a separate function named print_array that takes an array and an optional
# message as input, and prints them to the console. This function will be used to display
# the state of the array, maintaining separation from the manipulation logic.

def print_array(arr, msg = ""):
    print(msg, arr)



print("1. Array Creation:")
print("Create a one-dimensional NumPy array with values ranging from 1 to 10.")

one_dim_array_v1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print_array(one_dim_array_v1, "Answer v1: ")

one_dim_array_v2 = np.empty(10)
for i in range(10):
    one_dim_array_v2[i] = i+1
print_array(one_dim_array_v2, "Answer v2: ")
np.testing.assert_array_equal(one_dim_array_v2, one_dim_array_v1)

one_dim_array_v3 = np.arange(1, 11, 1)
print_array(one_dim_array_v3, "Answer v3: ")
np.testing.assert_array_equal(one_dim_array_v3, one_dim_array_v1)


print("Create a two-dimensional NumPy array (matrix) with shape (3, 3) containing values from 1 to 9.")

two_dim_array_v1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print_array(two_dim_array_v1, "Answer v1: ")

n = 3
two_dim_array_v2 = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        two_dim_array_v2[i, j] = n*i + j + 1
print_array(two_dim_array_v2, "Answer v2: ")
np.testing.assert_array_equal(two_dim_array_v2, two_dim_array_v1)

two_dim_array_v3 = np.zeros((n, n))
for i in range(n):
    two_dim_array_v3[i] = np.arange(i*n+1, (i+1)*n+1, 1)
print_array(two_dim_array_v3, "Answer v3: ")
np.testing.assert_array_equal(two_dim_array_v3, two_dim_array_v1)


print("2. Basic Operations:")

print("Indexing and Slicing:")
print("1. Access and print the third element of the one-dimensional array.")
third_element = one_dim_array_v3[2]
print(third_element, "Answer:")
np.testing.assert_equal(third_element, 3)

print("2.Slice and print the first two rows and columns of the two-dimensional array.")
two_dim_sliced = two_dim_array_v2[0:2, 0:2]
print_array(two_dim_sliced, "Answer:")
np.testing.assert_equal(two_dim_sliced, np.array([[1, 2], [4, 5]]))

print("Basic Arithmetic:")
print("1. Add 5 to each element of the one-dimensional array and print the result.")
one_dim_array_plus_5_v1 = one_dim_array_v3 + 5
print_array(one_dim_array_plus_5_v1, "Answer v1: ")
np.testing.assert_array_equal(one_dim_array_plus_5_v1, np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15]))


def add5(a):
    return a + 5

one_dim_array_plus_5_v2 = np.vectorize(add5)(one_dim_array_v3)
print_array(one_dim_array_plus_5_v2, "Answer v2: ")
np.testing.assert_array_equal(one_dim_array_plus_5_v2, np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15]))


print("2. Multiply each element of the two-dimensional array by 2 and print the result.")

two_dim_array_x2 = two_dim_array_v3 * 2
print_array(two_dim_array_x2, "Answer : ")
np.testing.assert_array_equal(two_dim_array_x2, np.array([[2, 4, 6], [8, 10, 12], [14, 16, 18]]))