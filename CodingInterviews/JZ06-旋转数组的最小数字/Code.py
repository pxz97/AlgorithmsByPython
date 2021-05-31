class Solution:
    def minNumberInRotateArray(self, rotateArray):
        """
        旋转数组的最小数字
        :param rotateArray: List, 旋转数组, 例如: [3,4,5,1,2]
        :return: int, 旋转数组的最小元素(若数组大小为0, 则返回0)
        """
        if not rotateArray:
            return 0
        left, right = 0, len(rotateArray) - 1
        while left < right:
            mid = (left + right) // 2
            if rotateArray[mid] < rotateArray[right]:
                right = mid
            elif rotateArray[mid] > rotateArray[right]:
                left = mid + 1
            else:
                right -= 1
        return rotateArray[left]
