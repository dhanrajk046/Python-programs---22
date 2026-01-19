def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]

def insertion_sort(arr):
    n = len(arr)
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

# Example usage:
arr = [64, 34, 25, 12, 22, 11, 90]
print("Original array:", arr)

bubble_sorted = arr.copy()
bubble_sort(bubble_sorted)
print("Bubble Sorted:", bubble_sorted)

selection_sorted = arr.copy()
selection_sort(selection_sorted)
print("Selection Sorted:", selection_sorted)

insertion_sorted = arr.copy()
insertion_sort(insertion_sorted)
print("Insertion Sorted:", insertion_sorted)