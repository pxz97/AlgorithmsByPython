class Solution:
    def IsContinuous(self, numbers):
        """
        扑克牌顺子
        :param numbers: List, 抽取的五张牌
        :return: bool, 是否为顺子
        """
        if not numbers:
            return False
        numbers.sort()
        zerosCount = numbers.count(0)
        for i in range(zerosCount, len(numbers) - 1):
            if numbers[i] == numbers[i + 1]:
                return False
            zerosCount = zerosCount - (numbers[i + 1] - numbers[i]) + 1
            if zerosCount < 0:
                return False
        return True
