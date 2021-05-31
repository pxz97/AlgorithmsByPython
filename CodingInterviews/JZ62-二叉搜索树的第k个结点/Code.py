class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    count = 0

    def KthNode(self, pRoot, k):
        if not pRoot:
            return None
        left = self.KthNode(pRoot.left, k)
        self.count += 1
        if self.count == k:
            return pRoot
        right = self.KthNode(pRoot.right, k)
        if left:
            return left
        if right:
            return right

