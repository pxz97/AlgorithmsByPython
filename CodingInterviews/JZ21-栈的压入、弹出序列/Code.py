class Solution:
    def IsPopOrder(self, pushV, popV):
        """
        栈的压入、弹出序列
        :param pushV: 压栈序列
        :param popV: 弹出序列
        :return:
        """
        stack = []
        for v in pushV:
            stack.append(v)
            while stack and stack[-1] == popV[0]:
                stack.pop()
                popV.pop(0)
        return not stack







