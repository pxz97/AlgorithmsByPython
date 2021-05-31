class Solution:
    def hasPath(self, matrix, word):
        """
        矩阵中的路径
        :param matrix: List[List[int]], 二维矩阵
        :param word: str, 要查找路径的字符串
        :return: bool, 是否存在这个路径
        """
        def dfs(m, n, k):
            if not 0 <= m < len(matrix) or not 0 <= n < len(matrix[0]) or matrix[m][n] != word[k]:
                return False
            if k == len(word) - 1:
                return True
            matrix[m][n] = ""
            res = dfs(m - 1, n, k + 1) or dfs(m + 1, n, k + 1) or dfs(m, n - 1, k + 1) or dfs(m, n + 1, k + 1)
            matrix[m][n] = word[k]
            return res
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if dfs(i, j, 0):
                    return True
        return False
