class Solution:
    def Fibonacci(self, n):
        """
        斐波那契数列
        :param n: int, 第几项
        :return: int, 斐波那契数列的第n项
        """
        a, b = 0, 1
        if n < 2:
            return n
        for _ in range(n - 1):
            a, b = b, a + b
        return b
