def countingSort(arr):
    bucket_len = max(arr) + 1
    bucket = [0] * bucket_len
    sorted_index = 0
    arr_len = arr
    for i in range(arr_len):
        if not bucket[arr[i]]:
            bucket[arr[i]] = 0
