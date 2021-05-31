class Solution:
    def NumberOf1(self, n):
        """
        二进制中1的个数
        :param n: int, 输入的整数
        :return: int, 1的个数
        """
        if n < 0:
            n = n & 0xffffffff
        count = 0
        while n:
            count += n & 1
            n = n >> 1
        return count
