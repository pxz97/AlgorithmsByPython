class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def IsBalanced_Solution(self, pRoot):
        """
        平衡二叉树
        :param pRoot: TreeNode, 二叉树的根节点
        :return: bool, 二叉树是否为平衡二叉树
        """
        def recu(root):
            if not root:
                return 0
            left = recu(root.left)
            if left == -1:
                return -1
            right = recu(root.right)
            if right == -1:
                return -1
            return max(left, right) + 1 if abs(left - right) < 2 else -1
        return recu(pRoot) != -1
