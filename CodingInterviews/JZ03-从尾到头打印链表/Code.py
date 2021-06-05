class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def printListFromTailToHead(self, listNode):
        """
        从头到尾打印链表
        :param listNode: ListNode, 链表头结点
        :return: List[int], 从尾部到头部的列表值序列
        """
        return self.printListFromTailToHead(listNode.next) + [listNode.val] if listNode else []
