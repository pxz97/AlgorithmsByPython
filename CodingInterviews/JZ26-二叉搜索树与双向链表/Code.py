class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def Convert(self, pRootOfTree):
        """
        二叉搜索树与双向链表
        :param pRootOfTree:
        :return: 链表中第一个节点的指针
        """
        if not pRootOfTree:
            return None
        if not pRootOfTree.left and not pRootOfTree.right:
            return pRootOfTree
        left = self.Convert(pRootOfTree.left)  # pRootOfTree左子树构成的有序链表的头节点（最小值）
        temp = left
        # 找左子树链表的最大值
        while left and temp.right:
            temp = temp.right
        # 将pRootOfTree连在左子树链表的最后面
        if left:
            temp.right = pRootOfTree
            pRootOfTree.left = temp
        right = self.Convert(pRootOfTree.right)  # pRootOfTree右子树构成的有序链表的头节点（最小值）
        # 将pRootOfTree连在右子树链表的最前面
        if right:
            right.left = pRootOfTree
            pRootOfTree.right = right
        return left if left else pRootOfTree

