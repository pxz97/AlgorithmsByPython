class Solution:
    def replaceSpace(self, s):
        """
        替换空格
        :param s: str, 替换前的字符串
        :return: str, 替换后的字符串
        """
        return "".join("%20" if c == " " else c for c in s)
