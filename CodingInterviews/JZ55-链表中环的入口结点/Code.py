class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def EntryNodeOfLoop(self, pHead):
        """
        链表中环的入口结点
        :param pHead: ListNode, 链表的头节点
        :return: ListNode, 环的入口结点
        """
        if not pHead or not pHead.next:
            return None
        slow, fast = pHead, pHead
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                fast = pHead
                while fast != slow:
                    fast = fast.next
                    slow = slow.next
                return slow
        return None
