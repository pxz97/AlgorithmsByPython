def bubbleSort(alist):
    for i in range(len(alist) - 1, 0, -1):
        for j in range(i):
            if alist[j] > alist[j + 1]:
                alist[j], alist[j + 1] = alist[j + 1], alist[j]
    return alist


def shortBubbleSort(alist):
    """
    冒泡排序要遍历整个未排好的部分，如果剩下的部分已经排序好了，冒泡排序仍会遍历，这会导致“浪费式”的交换
    设定一个参数exchanges，判断一轮遍历中是否发生元素变换，如果没有变换，说明列表已经有序
    """
    for i in range(len(alist) - 1, 0, -1):
        exchanges = False
        for j in range(i):
            if alist[j] > alist[j + 1]:
                alist[j], alist[j + 1] = alist[j + 1], alist[j]
                exchanges = True
            if not exchanges:
                break
    return alist


array = [3, 44, 38, 5, 47, 15, 36, 26, 27, 2, 46, 4, 19, 50, 48]
print("Before sort:", array)
print("After sort:", bubbleSort(array))
