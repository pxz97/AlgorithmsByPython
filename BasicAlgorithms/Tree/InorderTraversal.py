def inorderTraversal_recur(root):
    if not root:
        return []
    return inorderTraversal_recur(root.left) + [root.val] + inorderTraversal_recur(root.right)


def inorderTraversal(root):
    stack = []
    res = []
    cur = root
    while stack or cur:
        if cur:
            stack.append(cur)
            cur = cur.left
        else:
            cur = stack.pop()
            res.append(cur.val)
            cur = cur.right
    return res
