class TreeLinkNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        self.next = None


class Solution:
    def GetNext(self, pNode):
        """
        二叉树的下一个结点
        :param pNode: 指定的结点
        :return: 指定结点中序遍历的下一个结点
        """
        if not pNode:
            return None
        if pNode.right:
            p = pNode.right
            while p.left:
                p = p.left
            return p
        while pNode.next:
            if pNode.next.left == pNode:
                return pNode.next
            pNode = pNode.next
