class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def Print(self, pRoot):
        """
        把二叉树打印成多行
        :param pRoot: TreeNode, 二叉树的根节点
        :return: List, 二叉树按层打印的值
        """
        res = []
        if not pRoot:
            return res
        que = [pRoot]
        while que:
            tmp = []
            for _ in range(len(que)):
                tRoot = que.pop(0)
                tmp.append(tRoot.val)
                if tRoot.left:
                    que.append(tRoot.left)
                if tRoot.right:
                    que.append(tRoot.right)
            res.append(tmp)
        return res

