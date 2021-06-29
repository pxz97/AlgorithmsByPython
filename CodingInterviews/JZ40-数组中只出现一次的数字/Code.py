class Solution:
    def FindNumsAppearOnce(self, array):
        """
        数组中只出现一次的两个数字
        :param array: List[int], 整型数组
        :return: List[int], 只出现一次的两个数字
        """
        res = 0
        for a in array:
            res ^= a
        index = 0
        while res & 1 == 0:
            res >>= 1
            index += 1
        m, n = 0, 0
        for a in array:
            if self.helper(a, index):
                m ^= a
            else:
                n ^= a
        return [m, n]

    def helper(self, num, ind):
        num >>= ind
        return num & 1 != 0
