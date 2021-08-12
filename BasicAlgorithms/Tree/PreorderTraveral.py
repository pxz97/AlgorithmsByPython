def preorderTraversal_recur(root):
    if not root:
        return []
    return [root.val] + preorderTraversal_recur(root.left) + preorderTraversal_recur(root.right)


def preorderTraversal(root):
    
