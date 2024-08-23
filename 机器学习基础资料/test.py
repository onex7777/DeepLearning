import math


class ListNode():
    def __init__(self, x=0, next=None):
        self.val = x
        self.next = next

def construct_link(nums):
    curr = dummy = ListNode(0)
    if not nums:
        return dummy.next
    for num in nums:
        curr.next = ListNode(num)
        curr = curr.next
    return dummy.next

def reverseLink(head):
    pre, curr = None, head
    while curr is not None:
        tmp = curr.next
        curr.next = pre
        pre = curr
        curr = tmp
    return pre


def A(head):
    res = []
    while head:
        res.append(head.val)
        head = head.next
    return res

nums = list(range(1,5))
head = construct_link(nums)
head1 = reverseLink(head)
print(nums)
print(A(head1))


