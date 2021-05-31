class Solution:
    def jumpFloor(self, number):
        """
        跳台阶
        :param number: int, 跳number个台阶
        :return: int, 跳number个台阶有几种跳法
        """
        a, b = 1, 1
        if number < 2:
            return 1
        for _ in range(number - 1):
            a, b = b, a + b
        return b
