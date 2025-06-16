import numpy as np

# task-0
arr = np.random.randint(1, 51, size=(5, 4))

# task-1
print(arr)
anti_diag = []
for i in range(min(arr.shape)):
    anti_diag.append(arr[i, 3 - i])

print(anti_diag)

# task-2
max_in_rows = arr.max(axis = 1)
print(max_in_rows)

# task-3
overall_mean = arr.mean()
print(overall_mean)
new_arr = arr[arr <= overall_mean]
print(new_arr)

#task-4
def numpy_boundary_traversal(arr):
    m, n = arr.shape
    border_elements = []
    for i in range(n):
        border_elements.append(arr[0, i])
    for j in range(1, m):
        border_elements.append(arr[j, n - 1])
    if m > 1:
        for i in range(n - 2, -1, -1):
            border_elements.append(arr[m - 1, i])
    if n > 1:
        for j in range(m - 2, 0, -1):
            border_elements.append(arr[j, 0])

    return border_elements

print(numpy_boundary_traversal(arr))

