def quickSort(alist):
    quickSortHelper(alist, 0, len(alist) - 1)
    return alist


def quickSortHelper(alist, left, right):
    if left < right:
        mid = partition(alist, left, right)
        quickSortHelper(alist, left, mid - 1)
        quickSortHelper(alist, mid + 1, right)


def partition(alist, left, right):
    temp = alist[left]
    while left < right:
        while left < right and alist[right] > temp:
            right -= 1
        alist[left] = alist[right]
        while left < right and alist[left] < temp:
            left += 1
        alist[right] = alist[left]
    alist[left] = temp
    return left


array = [3, 44, 38, 5, 47, 15, 36, 26, 27, 2, 46, 4, 19, 50, 48]
print("Before sort:", array)
print("After sort:", quickSort(array))
