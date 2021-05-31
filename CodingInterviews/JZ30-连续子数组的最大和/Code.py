class Solution:
    def FindGreatestSumOfSubArray(self, array):
        """
        连续子数组的最大和
        :param array: List, 数组
        :return: int, 所有子数组和的最大值
        """
        if not array:
            return None
        for i in range(1, len(array)):
            array[i] = max(array[i - 1], 0) + array[i]
        return max(array)
