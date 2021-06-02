def selectionSort(alist):
    for i in range(len(alist) - 1):
        min_index = i
        for j in range(i + 1, len(alist) - 1):
            if alist[j] < alist[min_index]:
                min_index = j
        alist[i], alist[min_index] = alist[min_index], alist[i]
    return alist


array = [3, 44, 38, 5, 47, 15, 36, 26, 27, 2, 46, 4, 19, 50, 48]
print("Before sort:", array)
print("After sort:", selectionSort(array))
