class Solution:
    """
    用两个栈实现队列
    """
    def __init__(self):
        self.a = []
        self.b = []

    def push(self, node):
        self.a.append(node)

    def pop(self):
        if self.b:
            return self.b.pop()
        elif not self.a:
            return None
        else:
            while self.a:
                self.b.append(self.a.pop)
            return self.b.pop()