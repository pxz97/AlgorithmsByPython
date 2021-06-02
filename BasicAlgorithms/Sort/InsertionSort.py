def insertSort(alist):
    for i in range(1, len(alist)):
        current_value = alist[i]
        position = i
        while alist[position - 1] > current_value and position > 0:
            alist[position] = alist[position - 1]
            position = position - 1
        alist[position] = current_value
    return alist


array = [3, 44, 38, 5, 47, 15, 36, 26, 27, 2, 46, 4, 19, 50, 48]
print("Before sort:", array)
print("After sort:", insertSort(array))
