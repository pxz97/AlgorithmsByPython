class Solution:
    def rectCover(self, number):
        """
        矩形覆盖
        :param number: int, 一个2*number的大矩形由number个小矩形覆盖
        :return: int, 总共有几种覆盖方法
        """
        if number <= 2:
            return number
        a, b = 1, 2
        for _ in range(number - 1):
            a, b = b, a + b
        return a
