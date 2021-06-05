class Solution:
    def Power(self, base, exponent):
        """
        数值的整数次方
        :param base: float, 底数
        :param exponent: int, 指数
        :return: int, base的exponent次方
        """
        if base == 0:
            return 0
        if exponent < 0:
            base, exponent = 1 / base, -exponent
        res = 1
        while exponent:
            if exponent & 1:
                res *= base
            base *= base
            exponent >>= 1
        return res
