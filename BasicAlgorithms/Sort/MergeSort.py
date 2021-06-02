def mergeSort(alist):
    if len(alist) > 1:
        mid = len(alist) // 2
        left_half = alist[: mid]
        right_half = alist[mid:]

        left_half = mergeSort(left_half)
        right_half = mergeSort(right_half)

        i = 0
        j = 0
        k = 0
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                alist[k] = left_half[i]
                i = i + 1
            else:
                alist[k] = right_half[j]
                j = j + 1
            k = k + 1

        while i < len(left_half):
            alist[k] = left_half[i]
            i = i + 1
            k = k + 1

        while j < len(right_half):
            alist[k] = right_half[j]
            j = j + 1
            k = k + 1
    return alist


array = [3, 44, 38, 5, 47, 15, 36, 26, 27, 2, 46, 4, 19, 50, 48]
print("Before sort:", array)
print("After sort:", mergeSort(array))
