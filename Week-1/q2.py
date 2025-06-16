import numpy as np

# task-0
arr = np.random.uniform(0, 10, 20)

# task-1
print(arr)
rounded_arr = np.round(arr, 2)
print(rounded_arr)

# task-2
print(np.min(rounded_arr))
print(np.max(rounded_arr))
print(np.median(rounded_arr))

# task-3
modified_arr = np.where(rounded_arr < 5, np.round(rounded_arr**2, 2), rounded_arr)
print(modified_arr)

# task-4
def numpy_alternate_sort(array):
    sorted_arr = np.sort(array)
    result = []

    i, j = 0, len(sorted_arr) - 1
    while i <= j:
        result.append(sorted_arr[i])
        i += 1
        if i <= j:
            result.append(sorted_arr[j])
            j -= 1
    return np.array(result)

alt_sorted = numpy_alternate_sort(modified_arr)
print(alt_sorted)
