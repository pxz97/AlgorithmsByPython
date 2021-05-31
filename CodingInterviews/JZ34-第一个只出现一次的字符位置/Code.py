class Solution:
    def FirstNotRepeatingChar(self, s):
        """
        第一个只出现一次的字符
        :param s: str, 字符串
        :return: int, 第一次只出现一次的字符的位置
        """
        dic = {}
        for ss in s:
            dic[ss] = ss not in dic
        for ss in s:
            if dic[ss]:
                return s.index(ss)
        return -1
