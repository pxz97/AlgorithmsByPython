class Solution:
    def jumpFloorII(self, number):
        """
        跳台阶扩展问题
        :param number: int, 跳的台阶数
        :return: int, 跳number个台阶有几种跳法
        """
        if number <= 0:
            return 0
        res = 1
        for _ in range(number - 1):
            res *= 2
        return res