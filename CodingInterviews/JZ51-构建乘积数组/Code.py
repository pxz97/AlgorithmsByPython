class Solution:
    def multiply(self, A):
        """
        构建乘积数组
        :param A: List, 数组
        :return: List, 乘积数组
        """
        B, tmp = [1] * len(A), 1
        for i in range(1, len(A)):
            B[i] = B[i - 1] * A[i - 1]
        for i in range(len(A) - 2, -1, -1):
            tmp *= A[i + 1]
            B[i] *= tmp
        return B
