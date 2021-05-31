class Solution:
    def duplicate(self, numbers):
        """
        数组中重复的数字
        :param numbers: List, 数组
        :return: int, 重复的数字
        """
        n = 0
        while n < len(numbers):
            if numbers[n] == n:
                n += 1
                continue
            if numbers[numbers[n]] == numbers[n]:
                return numbers[n]
            numbers[numbers[n]], numbers[n] = numbers[n], numbers[numbers[n]]
        return -1
