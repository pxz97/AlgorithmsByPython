class Solution:
    def Find(self, target, array):
        """
        二维数组中的查找
        :param target: List[List[int]], 二维数组
        :param array: int, 目标值
        :return: bool, 是否含有该函数
        """
        if not array:
            return False
        row, col = len(array) - 1, 0
        while row >= 0 and col < len(array[0]):
            if array[row][col] < target:
                col += 1
            elif array[row][col] > target:
                row -= 1
            else:
                return True
        return False
