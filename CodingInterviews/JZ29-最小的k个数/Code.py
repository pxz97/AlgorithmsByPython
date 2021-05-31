import heapq


class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        """
        最小的K个数
        :param tinput: List, 给定的数组
        :param k: int, 最小数组的长度
        :return: List, 一个长度为k的最小数的数组
        """
        if k > len(tinput) or k == 0:
            return []
        hp = [-x for x in tinput[: k]]
        heapq.heapify(hp)
        for i in range(k, len(tinput)):
            if tinput[i] < - hp[0]:
                heapq.heapreplace(hp, -tinput[i])
        res = [-x for x in hp]
        res = sorted(res)
        return res
