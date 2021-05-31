class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def Mirror(self, pRoot):
        """
        二叉树的镜像
        :param pRoot: TreeNode, 原二叉树的头节点
        :return: TreeNode, 镜像操作后二叉树的头节点
        """
        if not pRoot:
            return None
        pRoot.left, pRoot.right = self.Mirror(pRoot.right), self.Mirror(pRoot.left)
        return pRoot
