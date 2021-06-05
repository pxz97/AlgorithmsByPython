class Solution:
    def reOrderArray(self, array):
        """
        调整数组顺序使奇数位于偶数前面
        :param array: List[int], 整数数组
        :return: List[int], 奇数位于前半部分，偶数位于后半部分，且奇与奇、偶与偶相对位置不变
        """
        odd, even = [], []
        for a in array:
            if a % 2 == 1:
                odd.append(a)
            elif a % 2 == 0:
                even.append(a)
        return odd + even
