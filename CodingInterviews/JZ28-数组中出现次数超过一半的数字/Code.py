class Solution:
    def MoreThanHalfNum_Solution(self, numbers):
        """
        数组中出现次数超过一半的数字
        :param numbers: List, 数组
        :return: int, 超过一半的数字
        """
        vote = 0
        for n in numbers:
            if vote == 0:
                tmp = n
            vote += 1 if n == tmp else -1
        return tmp
