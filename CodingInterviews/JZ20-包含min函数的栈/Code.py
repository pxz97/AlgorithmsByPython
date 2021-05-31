class Solution:
    def __init__(self):
        self.Stack = []
        self.minStack = []

    def push(self, node):
        self.Stack.append(node)
        if not self.minStack or node < self.minStack[-1]:
            self.minStack.append(node)

    def pop(self):
        if self.Stack[-1] == self.minStack[-1]:
            self.minStack.pop()
            return self.Stack.pop()
        else:
            return self.Stack.pop()

    def top(self):
        return self.Stack[-1]

    def min(self):
        if not self.minStack:
            return None
        return self.minStack[-1]