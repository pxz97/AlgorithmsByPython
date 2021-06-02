def shellSort(alist):
    gap = len(alist) // 2
    while gap > 0:
        for i in range(gap):
            gapInsertionSort(alist, i, gap)
        gap = gap // 2
    return alist


def gapInsertionSort(alist, start, gap):
    for i in range(start + gap, len(alist), gap):
        position = i
        current_value = alist[i]

        while position > start and alist[position - gap] > current_value:
            alist[position] = alist[position - gap]
            position = position - gap
        alist[position] = current_value


array = [3, 44, 38, 5, 47, 15, 36, 26, 27, 2, 46, 4, 19, 50, 48]
print("Before sort:", array)
print("After sort:", shellSort(array))
