class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def ReverseList(self, pHead):
        """
        反转链表
        :param pHead: ListNode, 链表的头节点
        :return: ListNode, 反转后链表的头节点
        """
        cur, pre = pHead, None
        while cur:
            tmp = cur.next
            cur.next = pre
            pre = cur
            cur = tmp
        return cur
