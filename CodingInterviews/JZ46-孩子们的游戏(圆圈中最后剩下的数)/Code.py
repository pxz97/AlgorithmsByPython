def LastRemaining_Solution(n, m):
    # write code here
    nlist = list(range(n))
    while nlist:
        count = 0
        for i, num in enumerate(nlist):
            count += 1
            if count == m:
                nlist.pop(i)
                continue
    return num

print(LastRemaining_Solution(5, 3))