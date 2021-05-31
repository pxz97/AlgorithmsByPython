import math


class Solution:
    def cutRope(self, number):
        """
        剪绳子
        :param number: int, 绳子长度
        :return: 剪成多段后，最大的乘积
        """
        if number <= 3:
            return number - 1
        a, b = number // 3, number % 3
        if b == 0:
            return math.pow(3, a)
        elif b == 1:
            return math.pow(3, a - 1) * 4
        else:
            return math.pow(3, a) * 2
