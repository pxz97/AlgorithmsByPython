class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def Merge(self, pHead1, pHead2):
        """
        合并两个排序的链表
        :param pHead1: ListNode, 第一个链表的头节点
        :param pHead2: ListNode, 第二个链表的头节点
        :return: ListNode, 合并后链表的头节点
        """
        cur = head = ListNode(0)
        while pHead1 and pHead2:
            if pHead1.val < pHead2.val:
                cur.next, pHead1 = pHead1, pHead1.next
            else:
                cur.next, pHead2 = pHead2, pHead2.next
            cur = cur.next
        cur.next = pHead1 if pHead1 else pHead2
        return head.next


