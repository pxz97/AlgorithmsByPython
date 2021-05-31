class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def FindFirstCommonNode(self, pHead1, pHead2):
        """
        两个链表的第一个公共结点
        :param pHead1: ListNode, 第一个无环链表
        :param pHead2: ListNode, 第二个无环链表
        :return: ListNode, 第一个公共结点
        """
        p1, p2 = pHead1, pHead2
        while p1 != p2:
            p1 = p1.next if p1 else pHead2
            p2 = p2.next if p2 else pHead1
        return p1
